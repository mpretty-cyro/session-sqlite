#pragma once

#include <SQLiteCpp/Database.h>
#include <SQLiteCpp/SQLiteCpp.h>

#include <chrono>
#include <concepts>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iterator>
#include <mutex>
#include <optional>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <type_traits>
#include <variant>
#include <vector>

#include "secure_buffer.hpp"

namespace session::sqlite {

class blob_size_error : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

/// A bind argument that binds each element of a sequence to a consecutive parameter, rather than
/// binding the sequence as one value.
///
/// This is the bind-side counterpart of `placeholders()`, and exists for the one query shape whose
/// parameter count is not known until runtime:
///
///     auto query = "SELECT ... WHERE id IN ({})"_format(placeholders(ids.size()));
///     for (auto&& [a, b] : conn.prepared_results<int, int>(query, bind_each{ids}))
///
/// Without it a variable-length `IN` list cannot go through the normal bind path at all -- the
/// parameters are a template pack, and a runtime-sized sequence cannot be expanded into one -- which
/// leaves a caller hand-binding a `prepared_st` in a loop and losing `prepared_results` with it.
///
/// Takes a container (or view), or an iterator pair for binding part of one:
///
///     conn.prepared_exec(query, before, bind_each{ids}, after);
///     conn.prepared_exec(query, bind_each{ids.begin(), ids.begin() + n});
///
/// It consumes as many parameters as the sequence has elements, and the parameters around it number
/// from wherever it leaves off, so it composes with ordinary bind values on either side and with
/// more than one of itself.  **The count must match the placeholders in the query**: too few and
/// the rest bind as NULL, too many and SQLite throws.  Deriving both from the same sequence is what
/// keeps them in step.
///
/// Holds iterators, and binds strings without copying as everywhere else here, so the sequence and
/// whatever its elements reference must both outlive the statement's execution.  Intended to be
/// written inline in the bind call, where that is automatic.
template <std::input_iterator It, std::sentinel_for<It> End = It>
struct bind_each {
    It first;
    End last;

    bind_each(It first, End last) : first{std::move(first)}, last{std::move(last)} {}

    template <std::ranges::input_range R>
    bind_each(const R& values) :
            first{std::ranges::begin(values)}, last{std::ranges::end(values)} {}
};

template <std::ranges::input_range R>
bind_each(const R&)
        -> bind_each<std::ranges::iterator_t<const R>, std::ranges::sentinel_t<const R>>;

template <std::input_iterator It, std::sentinel_for<It> End>
bind_each(It, End) -> bind_each<It, End>;

// Decorating for extracting BLOB values from a query without unnecessary copying.  This is intended
// to be called via `db::get` such as:
//
//    auto [num, data] = db::get<int, blob>(st);
//
// Where `data` will be a `blob` (which is an empty wrapper around std::span) that points to the
// BLOB data directly.  Note that this view remains only value while the statement remains active,
// and so it must be used as needed immediately.
//
// Note that the lifetime limitation above means that this is *unsuitable* for one-shot methods like
// `prepared_get/prepared_maybe_get` as they finalize the statement before returning the value.
//
// To *bind* input values as BLOBs, simply pass the value as a std::byte or unsigned char std::span
// to the prepared_exec and similar functions.
//
// When using blobn<N> with fixed N then you get an extended fixed-Extent span; this will throw an
// exception if the returned value does not match: it is generally recommended to only be used when
// the database schema or query conditions already ensure the length.
template <size_t Extent = std::dynamic_extent>
struct blobn : std::span<const std::byte, Extent> {
    blobn(SQLite::Column&& col) :
            std::span<const std::byte, Extent>{
                    static_cast<const std::byte*>(col.getBlob()), [&](size_t len) {
                        if constexpr (Extent != std::dynamic_extent)
                            if (len != Extent)
                                throw blob_size_error{
                                        "Unable to extract BLOB: expected " +
                                        std::to_string(Extent) + " bytes, got " +
                                        std::to_string(len)};
                        return len;
                    }(col.getBytes())} {}
};
using blob = blobn<std::dynamic_extent>;

// True for a blob view, and for anything that merely wraps one: an optional<blob> is the same
// hazard, since wrapping a span does not make it outlive the statement it points into.
template <typename T>
constexpr bool holds_blob_view = false;
template <size_t Extent>
constexpr bool holds_blob_view<blobn<Extent>> = true;
template <typename T>
constexpr bool holds_blob_view<std::optional<T>> = holds_blob_view<T>;

// Takes a trivial, no-padding struct from which we can directly initialize from the (fixed size)
// stored blob value.  The type `T` must be a trivially copyable type.  Unlike `blob` this value
// *is* suitable for use in a one-shot method as the blob value is copied into the `T value` on
// extraction.
//
// For example:
//
//     auto x = conn.prepared_get<std::array<std::byte, 32>>("SELECT key FROM t1 WHERE id = ?", 42);
//
// The returned type is an empty wrapper around T.
template <typename T>
    requires std::has_unique_object_representations_v<T> &&
             std::is_trivially_default_constructible_v<T>
struct blob_guts : T {
    blob_guts(SQLite::Column&& col) {
        blob b{std::move(col)};
        if (b.size() != sizeof(*this))
            throw blob_size_error{
                    "Unable to extract DB BLOB value into type: wrong data size (" +
                    std::to_string(b.size()) + " vs " + std::to_string(sizeof(*this)) + ")"};
        std::memcpy(this, b.data(), b.size());
    }
};

namespace detail {
    template <typename T>
    constexpr bool is_optional = false;
    template <typename T>
    constexpr bool is_optional<std::optional<T>> = true;

    template <typename T, typename... More>
    struct first_type {
        using type = T;
    };
    template <typename... T>
    using first_type_t = typename first_type<T...>::type;

    template <typename... T>
    using type_or_tuple =
            std::conditional_t<sizeof...(T) == 1, first_type_t<T...>, std::tuple<T...>>;

    // Binds anything convertible to string_view (c strings, std::string, string_view itself).
    inline void bind_oneshot_single(SQLite::Statement& st, int i, std::string_view val) {
        st.bindNoCopy(i, val);
    }
    // Binds something convertible to a std::span of bytes or unsigned chars.  This allows you to
    // bind things like `std::array<std::byte, 32>` of vectors of unsigned char, which will get
    // converted to BLOBs.
    inline void bind_oneshot_single(SQLite::Statement& st, int i, std::span<const std::byte> val) {
        st.bindNoCopy(i, static_cast<const void*>(val.data()), val.size());
    }
    inline void bind_oneshot_single(
            SQLite::Statement& st, int i, std::span<const unsigned char> val) {
        st.bindNoCopy(i, static_cast<const void*>(val.data()), val.size());
    }
    // Binds a std::monostate as a NULL value.  std::monostate is intended for use as a "not set"
    // value in a variant as a more compact alternative to std::optional<std::variant<...>>.
    inline void bind_oneshot_single(SQLite::Statement& st, int i, const std::monostate) {
        st.bind(i);
    }

    // nullptr_t becomes NULL
    inline void bind_oneshot_single(SQLite::Statement& st, int i, std::nullptr_t) {
        st.bind(i);
    }

    // Bind integers, dispatching to a 32- or 64-bit SQLite bind by size.  This is a single
    // constrained template rather than fixed-width (int32_t/int64_t) overloads because the exact
    // fundamental type behind a fixed-width alias varies by platform: e.g. on LP64 targets int64_t
    // is `long`, so a `long long` argument (such as libc++'s chrono duration rep) matches none of
    // the fixed widths exactly and is ambiguous.  Unsigned 64-bit values are excluded here (and
    // rejected via the deleted overload below) because SQL does not support values larger than max
    // int64_t.
    template <std::integral T>
        requires (std::signed_integral<T> || sizeof(T) <= sizeof(uint32_t))
    void bind_oneshot_single(SQLite::Statement& st, int i, T val) {
        if constexpr (std::unsigned_integral<T>)
            st.bind(i, static_cast<uint32_t>(val));
        else if constexpr (sizeof(T) <= sizeof(int32_t))
            st.bind(i, static_cast<int32_t>(val));
        else
            st.bind(i, static_cast<int64_t>(val));
    }
    inline void bind_oneshot_single(SQLite::Statement& st, int i, double val) {
        st.bind(i, val);
    }
    // Reject unsigned values wider than 32 bits (uint64_t and friends): SQL does not support values
    // larger than max int64_t.  A constrained template (rather than a fixed uint64_t overload)
    // catches every such type -- uint64_t, unsigned long, unsigned long long -- on every platform.
    template <std::unsigned_integral T>
        requires (sizeof(T) > sizeof(uint32_t))
    void bind_oneshot_single(SQLite::Statement& st, int i, T val) = delete;

    // Binds a scoped enum as its underlying type, so that a type-safe integer enum can be bound
    // directly rather than having to be cast at every call site.
    template <typename T>
        requires std::is_enum_v<T>
    void bind_oneshot_single(SQLite::Statement& st, int i, T val) {
        bind_oneshot_single(st, i, static_cast<std::underlying_type_t<T>>(val));
    }

    // Binds an optional<T>: if val is not set this binds a SQL NULL value, otherwise it recurses to
    // bind whatever the value is.
    template <typename T>
    void bind_oneshot_single(SQLite::Statement& st, int i, const std::optional<T>& val) {
        if (val)
            bind_oneshot_single(st, i, *val);
        else
            st.bind(i);  // binds NULL
    }
    // Binds a variant by binding whatever value the variant has.  Thus you can bind, e.g., a
    // `variant<monostate, int, std::string>` to bind either a NULL, INTEGER, or TEXT value.  Each
    // possible alternative must be something bindable.
    template <typename... T>
    void bind_oneshot_single(SQLite::Statement& st, int i, const std::variant<T...>& val) {
        std::visit([&st, i](const auto& val) { bind_oneshot_single(st, i, val); }, val);
    }

    // Binds one argument, advancing `i` past whatever parameters it consumed.  One for an ordinary
    // value; as many as it has elements for a `bind_each`, which is why the parameter number is a
    // running counter rather than the argument's position: with a sequence in the pack the two stop
    // being the same thing.
    template <typename T>
    void bind_oneshot_arg(SQLite::Statement& st, int& i, const T& val) {
        bind_oneshot_single(st, i++, val);
    }

    template <std::input_iterator It, std::sentinel_for<It> End>
    void bind_oneshot_arg(SQLite::Statement& st, int& i, const bind_each<It, End>& each) {
        for (auto it = each.first; it != each.last; ++it)
            bind_oneshot_single(st, i++, *it);
    }

    template <typename... T>
    void bind_oneshot(SQLite::Statement& st, const T&... bind) {
        // maybe_unused because the fold is empty when there is nothing to bind, which is a
        // perfectly ordinary call (e.g. a query with no parameters).
        [[maybe_unused]] int i = 1;
        (bind_oneshot_arg(st, i, bind), ...);
    }
}  // namespace detail

// The following methods are helpers that operate on an SQLite::Statement, and while they can be
// used with a direct SQLite::Statement, they are primarily intended to be used in combination with
// a StatementWrapper (which is implicitly convertible to SQLite::Statement) constructed via
// conn.prepared_st(...).

// Called from exec_query and similar to bind statement parameters for immediate execution.
// string/string_views (and c strings) are bound as TEXT values using no-copy binding; integer and
// double values are bound by value; nullptr binds to NULL, std::optional binds to either NULL (if
// empty) or the contained value; variant binds whichever variant value is held.  std::spans of
// std::byte or unsigned char spans (or types convertible to them) bind as BLOBs containing the
// contained value.
// A `bind_each` argument binds each of its elements to a consecutive parameter; see there.
template <typename... T>
void bind_oneshot(SQLite::Statement& st, const T&... bind) {
    detail::bind_oneshot(st, bind...);
}

// Executes a query that does not expect results.  Optionally binds parameters, if provided.
// Returns the number of affected rows; throws on error or if results are returned.
template <typename... T, int... Index>
int exec_query(SQLite::Statement& st, const T&... bind) {
    bind_oneshot(st, bind...);
    return st.exec();
}

// Same as above, but prepares a literal query on the fly for use with queries that are only used
// once.
template <typename... T>
int exec_query(SQLite::Database& db, const std::string& query, const T&... bind) {
    SQLite::Statement st{db, query};
    return exec_query(st, bind...);
}

// Retrieves a single row of values from the current state of a statement (i.e. after a
// executeStep() call that is expecting a return value).  If `T...` is a single type then this
// returns the single T value; if T... has multiple types then you get back a tuple of values.
//
// std::optional<T> columns are handled specially: a SQL NULL column value produces std::nullopt
// rather than a default-constructed T (since SQLite::Column::operator T() returns 0/empty for
// NULL values, which would produce a non-empty optional{0} without this special handling).
namespace detail {
    // A scoped enum has to be converted via its underlying type: SQLite::Column offers conversion
    // operators for every fixed-width integer type, so casting one straight to an enum is
    // ambiguous rather than merely narrowing.
    template <typename T>
    T col_to(SQLite::Column&& col) {
        if constexpr (std::is_enum_v<T>)
            return static_cast<T>(static_cast<std::underlying_type_t<T>>(std::move(col)));
        else
            return static_cast<T>(std::move(col));
    }

    template <typename T, int I>
    T get_col(SQLite::Statement& st) {
        if constexpr (is_optional<T>) {
            auto col = st.getColumn(I);
            if (col.isNull())
                return std::nullopt;
            return std::optional{col_to<typename T::value_type>(std::move(col))};
        } else {
            return col_to<T>(st.getColumn(I));
        }
    }
    template <typename... T, int... Is>
    std::tuple<T...> get_tuple(SQLite::Statement& st, std::integer_sequence<int, Is...>) {
        return {get_col<T, Is>(st)...};
    }
}  // namespace detail
template <typename T>
T get(SQLite::Statement& st) {
    return detail::get_col<T, 0>(st);
}
template <typename T1, typename T2, typename... Tn>
std::tuple<T1, T2, Tn...> get(SQLite::Statement& st) {
    return detail::get_tuple<T1, T2, Tn...>(
            st, std::make_integer_sequence<int, 2 + sizeof...(Tn)>{});
}

// Steps a statement to completion that is expected to return at most one row, optionally binding
// values into it (if provided).  Returns a filled out optional<T> (or optional<std::tuple<T...>>)
// if a row was retrieved, otherwise a nullopt.  Throws if more than one row is retrieved.
template <typename... T, typename... Args>
std::optional<detail::type_or_tuple<T...>> exec_and_maybe_get(
        SQLite::Statement& st, const Args&... bind) {
    static_assert(
            !(holds_blob_view<T> || ...),
            "blobn<N>/blob cannot be used as a return type for exec_and_maybe_get/exec_and_get/"
            "prepared_get/prepared_maybe_get, on its own or inside an optional: the returned span "
            "is invalidated when the statement is finalized. Use blob_guts<T> instead to copy the "
            "data out.");
    bind_oneshot(st, bind...);
    std::optional<detail::type_or_tuple<T...>> result;
    while (st.executeStep()) {
        if (result)
            throw std::runtime_error{"DB error: expected single-row result, got multiple rows"};
        result = get<T...>(st);
    }
    return result;
}

// Executes a statement to completion that is expected to return exactly one row, optionally
// binding values into it (if provided).  Returns a T or std::tuple<T...> (depending on whether or
// not more than one T is provided) for the row.  Throws an exception if no rows or more than one
// row are returned.
template <typename... T, typename... Args>
detail::type_or_tuple<T...> exec_and_get(SQLite::Statement& st, const Args&... bind) {
    auto maybe_result = exec_and_maybe_get<T...>(st, bind...);
    if (!maybe_result)
        throw std::runtime_error{"DB error: expected single-row result, got no rows"};
    return *std::move(maybe_result);
}

// Executes a query to completion, collecting each row into a vector<T> (or vector<tuple<T...>> if
// multiple T are given).  Can optionally bind before executing.
template <typename... T, typename... Bind>
std::vector<detail::type_or_tuple<T...>> get_all(SQLite::Statement& st, const Bind&... bind) {
    bind_oneshot(st, bind...);
    std::vector<detail::type_or_tuple<T...>> results;
    while (st.executeStep())
        results.push_back(get<T...>(st));
    return results;
}

/// Supported encryption types by this library.  These may not all be supported, depending on the
/// specific sqlite3 backend in use.  You can check `enabled(Encryption::Type)` to see if the
/// current backend is compiled with support for the given encryption type.
///
/// If you aren't sure what to choose:
/// - Use ChaCha20 if you need something that is reasonably fast everywhere, and you are using a
///   random 32-byte key.
/// - Use AEGIS256 if you want something extra fast on modern AMD/Intel CPUs (with AES-NI
///   instructions) or ARM64 CPUs (with AES crypto extensions).
/// - Do not use the built-in sqlcipher or sqlite3mc plaintext password support unless you
///   absolutely must produce databases that other libraries need.  Instead, if you have a secure
///   random 32-byte value then just use that.  If you have a user-supplied password then salt and
///   hash a plaintext password yourself via Argon2id, and pass that as a raw key.  See
///   plaintext_password for more on why.
///
enum class Encryption {
    /// No encryption.  The database will be a plain sqlite3 database readable by any standard
    /// sqlite3 tools.
    None,

    /// ChaCha20 with poly1305 per-page authentication tag, and 64007 kdf iterations when using a
    /// plaintext password.  This encryption mechanism requires that the sqlite3 implementation is
    /// actually sqlite3-multiple-ciphers and is the fastest, most secure choice for encryption
    /// without compromising security on general devices (at least when using a 32-byte key rather
    /// than a plaintext password).
    ///
    /// This reduces the effective size of every 4KiB on-disk page by 32 bytes (to store the 16-byte
    /// tag + 16-byte nonce), i.e. about 0.8% database size overhead.
    ///
    /// This is a strong, general purpose choice (at least when used with random keys rather than
    /// plaintext passwords) that runs well on both modern and older hardware.
    ChaCha20,

    /// AEGIS-256.  This is an modern AEAD with better performance on modern CPUs than AES or
    /// ChaCha20, with security margins somewhere in between the two (although none of the three are
    /// weak).
    ///
    /// If you are primarily targeting modern CPUs with AES-NI or ARM-crypto extensions (i.e.
    /// recent Intel/AMD or ARM CPUs) this is an excellent choice that will have noticeably lower
    /// CPU overhead than the other choices here.  It will, however, be slower on slower, older
    /// systems, and so you may want to prefer ChaCha20 if you are going to use the same encryption
    /// everywhere.
    ///
    /// Per-page Overhead is 64 bytes per 4kiB page, i.e. about 1.6% database size overhead.
    AEGIS256,

    /// SQLCipher4 standard, where encryption uses AES-256-CBC with a HMAC-SHA512 tag, and 256000
    /// kdf iterations when using a plaintext password.
    ///
    /// TL;DR: this is slower than ChaCha20 and AEGIS256 in all cases; only use this over one of
    /// those if you specifically need compatibility with an SQLCipher4 database.
    ///
    /// SQLCipher4's encryption and decryption is simply slower than the others, even with CPU AES
    /// hardware acceleration, because the vast majority of the encryption happens not in AES but in
    /// the per-page HMAC-SHA512 calculation.  *Just* the HMAC-SHA512 tag calculation consumes
    /// nearly double the CPU cycles of the combined chacha20+poly1305 encryption and tag
    /// calculation, and so even with highly accelerated AES, CPU time required for encryption is at
    /// least double ChaCha20.  Without hardware acceleration, performance is worse still because
    /// AES becomes much slower than ChaCha20, and so on CPUs lacking AES-NI or ARM-crypto
    /// extensions, total AES + HMAC-SHA512 cpu time is somewhere around 4-5x ChaCha20+poly1305.
    ///
    /// Additionally, if the backend sqlite3 implementation is actually sqlite-multiple-ciphers, the
    /// AES implementation used here is *always* software-based (as the library does not carry heavy
    /// dependencies like OpenSSL to perform AES, nor does it have AES-NI/ARM-crypto
    /// implementations), and so when the backend is sqlite-mc expect this to be about 4-5x the CPU
    /// overhead of ChaCha20.
    ///
    /// Per-page overhead is 80 bytes per 4kiB page, i.e. about 2.0% size overhead.
    SQLCipher4,

    /// SQLCipher3, which is AES-256-CBC with a HMAC-SHA1 tag and 64000 kdf iterations in plaintext
    /// password mode.
    ///
    /// This is slower than ChaCha20 and AEGIS256 in all cases, slightly slower than SQLCipher4 on
    /// 64-bit systems; and weaker security-wise than any of those.  Only use this to load an
    /// existing SQLCipher3 database.
    ///
    /// Storage overhead is 48 bytes per 4kiB page, i.e. about 1.2% storage size overhead.
    SQLCipher3,

    /// Ascon128 is a sqlite-multiple-ciphers supported AEAD designed to be very lightweight while
    /// still reasonably secure, using 64007 PBKDF2 rounds with an Ascon-derived hash function.
    /// This is primarily aimed at *extremely* limited devices such as IoT devices (e.g. ESP32,
    /// Cortex-M0 with avialable ram size in the KB range) and is not recommended unless you
    /// specifically need to target such a device (or read a database compatible with such a
    /// device).  This is generally slower than any of the other algorithms here, and shouldn't be
    /// used unless you know you need it.
    ///
    /// Disk page overhead is 32 bytes (same as ChaCha20).
    Ascon128,
};

/// Returns true if the given encryption type is supported by the sqlite3 backend in use.
bool enabled(Encryption type);

// The following argument types are designed to pass various values into the Database constructor
// that govern how the database is opened and which keys or settings are applied.
//
// These options can be passed in any order after the Encryption type.

/// `plaintext_header` is an argument value to pass to the Database constructor to create the
/// database in Apple iOS special snowflake mode that avoids encrypting the first 24/32 bytes of a
/// file so that Apple can apply different rules to the process (such as not immediately killing it)
/// because files that look SQLite database get special handling because this is the sort of
/// bullshit OS-level hacks that Apple thinks makes for good software design.
///
/// If you are using a raw key (as opposed to a user-supplied password) with one of the preferred
/// ciphers (AEGIS, ChaCha20, or Ascon128) then all you need to pass is this value.  If you are
/// using a password (either plaintext or wrapped with argon2id_password) then you must also specify
/// and store a salt yourself, and must pass it every time you open the database with the
/// `salt{...}` option.
///
/// For AEGIS, ChaCha20, and Ascon128 with raw keys, the salt is not used at all in plaintext
/// password mode, and is not needed to decrypt the database.
///
/// SQLCipher3/4 always requires a salt, and so even in raw key mode you must provide it when using
/// plaintext header mode.  (This is because SQLCipher does a completely pointless extra salted hash
/// round even when you use a raw key, which adds no security whatsoever but probably ticked a
/// security theatre checkbox somewhere that made one of their corporate customers pay^Whappy).
///
/// Because this salt does nothing (cryptographically), it is perfectly acceptable (and, in fact,
/// recommended) to use a fixed salt value with SQLCipher + raw keys so that the salt doesn't have
/// to be stored separately at all.  (Of course, if you are loading an existing, plaintext header
/// SQLCipher encrypted database, then you have to pass whatever the salt value it was created
/// with).
///
/// Depending on the encryption algorithm in use this will leave either the first 24 (everything
/// except SQLCipher3/4) or 32 (SQLCipher3/4) bytes of the file unencrypted.  (Only the first 24
/// need to be unencrypted to activate iOS special snowflake mode, but SQLCipher's AES-CBC block
/// encryption mode requires that the value is a multiple of the 16-byte encryption block size).
///
/// Using this leaks a tiny amount of metadata -- you can tell it's an encrypted SQLite file and,
/// with SQLCipher, it also leaks the file change counter (which lives in bytes 24-27).  (Bytes
/// 28-31 are the page count, which doesn't seem like much of a leak since you can figure this out
/// from the file size already).
///
/// If you are *not* using a plaintext header then you don't need to worry about the salt at all
/// (and thus there is no interface to pass it otherwise): it will be automatically read/stored from
/// the first 16 bytes of the file.
struct plaintext_header_t {};
inline constexpr plaintext_header_t plaintext_header;

/// Specifies a 16-byte database salt.  There are two main use cases for this:
/// - When using `plaintext_header` with a plaintext password: the salt is not stored in the
///   database and so must be stored by the application and provided each time the database is
///   opened.  (If not using plaintext_header, the salt is available as the first 16 bytes of the
///   database file).
/// - With SQLCipher with `plaintext_header`: even when using raw key, SQLCipher still requires the
///   salt to decrypt the database.  (AEGIS/ChaCha20/Ascon128 do not).
///
/// And one silly case:
/// - AEGIS/ChaCha20/Ascon128 vanity header (when using raw keys).  If, for some reason, you want to
///   control the first 16 bytes of the file then simply specify your desired 16-byte vanity prefix
///   when you first create it; it will get stored as the salt value at the beginning of the file,
///   but will never actually affect encryption when using raw keys.  DO NOT DO THIS with plaintext
///   password -- the security of the database with a plaintext password depends on using a random
///   salt!
///
/// If you are creating a new database and fit into one of the cases above, then:
/// - If you are using SQLCipher3/4 with a raw key, just use a fixed 16-byte salt, e.g. your
///   application name.
/// - If you are using a plaintext password, generate a random salt, for instance with sodium's
///   randombytes_buf.
/// - If you are making a vanity salt with AEGIS/ChaCha20/Ascon128 (and using raw keys) then have
///   fun.
struct salt {
    std::span<const std::byte, 16> salt;
};

/// Database constructor argument wrapper used to provide a 32-byte key to use for database
/// encryption.  The value will be copied into a secure memory (to be used with multiple connections
/// as needed).
struct raw_key {
    std::span<const std::byte, 32> key;
};

/// Database constructor argument wrapper used to pass a plaintext password for database encryption,
/// using the default password hashing algorithm for the chosen encryption type.
///
/// Using this is only recommended for opening existing encrypted databases that were created with a
/// built-in password.  If you are starting out from scratch, and still want to use a user-provided
/// password, see argon2id_password instead.
///
/// The built-in password hashing for every encryption layer other than AEGIS256 and Ascon128 should
/// be considered weak as they use PBKDF2, which scales extremely well on GPU clusters but quite
/// poorly on CPUs, and so the number of iterations that you need to actually be somewhat resistant
/// to brute force cracking under PBKDF2 makes the database opening time with a *legitimate*
/// password painfully slow.
///
/// Moreover, because we can only store the password, *every* connection created here (i.e. if using
/// simultaneous connections from multiple threads) will incur the overhead that they would not if
/// using a raw key.
///
/// TL;DR: avoid using this if at all possible.
///
/// As to *why* the plaintext password support is so bad: all of the encryption algorithms other
/// than AEGIS256 and Ascon128 use PBKDF2 with either HMAC-SHA256 of HMAC-SHA512 to hash a plaintext
/// password with a random salt through many thousands of rounds of hashing to make the final key
/// derivation costly to compute when used correctly with a random salt.
///
/// In practice, though, PBKDF2 is of questionable usefulness today.  It was standardized in 2000,
/// when an "attacker" was just someone with more CPUs (which wasn't that easy: this predates even
/// the first commercial dual core CPU), and the performance difference between a low-end CPU and
/// high-end CPU was maybe 5x.  Cracking a password would require lots of expensive CPUs running in
/// parallel: your quarter of a second (or whatever) of CPU time for the legitimate password would
/// have to be incurred on an attacker's CPUs for each guess of the password.  That is no longer the
/// reality today: the adversary is not running a CPU, they are running on GPUs that can compute
/// many thousands of PBKDF2 iterations in parallel at once on a single GPU, essentially meaning the
/// cracking device is thousands (or more) times more powerful, and can readily access GPU clusters
/// that can easily and cheaply compute millions of PBKDF2 hashes in the time your mobile CPU can
/// compute one.
///
/// The only way you can regain some security is by making your own life worse: cranking up the
/// number of iterations so high that you have a very noticeable, battery-burning, CPU fan spin up
/// to compute it so that it's still hopefully a little bit expensive still on the GPU cluster than
/// runs millions of times faster than your device, and just hope that someone doesn't want your
/// password too badly.
///
/// Argon2 was invented to help alleviate this problem: it is deliberately designed to not scale
/// well on GPUs by being a memory-hard algorithm, with memory requirements per hash that are not a
/// big deal on CPUs but are much, much higher than the thread-to-memory ratio of modern GPU
/// clusters.
///
/// And so, if you must use a user-provided plaintext password for encryption, roll your own salted
/// Argon2id hash with parameters suitable for your device and user tolerances and provide the
/// result as a raw key when opening the database, or use the argon2id_password wrapper provided
/// here. This will work with *any* of the encryption algorithms in raw key mode.
///
/// As for AEGIS256: it correctly uses Argon2id by default, but you are still recommended to
/// calculate it yourself (even with AEGIS256 defaults, if you want them) so that additional
/// connections needed for simultaneous thread access can avoid the hash recomputation.
///
/// In general, however, if you have some means of encrypting a secure-random 32-byte value,
/// generate and use that: you avoid all the overhead of password hashing and you have a key that is
/// as secure as a theoretical infinite round password hash.
struct plaintext_password {
    std::string_view pass;
};

/// This constructor arguments instructs the Database constructor to compute a salted Argon2id hash
/// of the given plaintext password and use that hash as the raw key mode.  The default values for
/// opslimit and memlimit are chosen to produce exactly the same result as the plaintext AEGIS256
/// password encryption is perhaps on the low side; you may want to increase them, except on iOS
/// because Apple likes devs to suffer with insufferable memory limits for extension processes (this
/// suffering is part of the Apple Stockholm syndrome otherwise known as the Apple walled garden).
///
/// When this option is combined with plaintext_header, the salt will be taken from the value in
/// plaintext_header.  Otherwise, if the file exists, it will be read from the first 16 bytes of the
/// file (which is where sqlite3mc and sqlcipher store the salt), and otherwise it will be randomly
/// generated and provided during database creation.
///
/// In other words:
/// - if using plaintext header, generate a random salt and pass it to the plaintext_header option.
///   The salt for argon2 will use that value.  You will need to store that salt value externally
///   (but it is perfectly fine to store it unencrypted).
/// - if not using plaintext header then don't worry about the salt at all: it will be written to
///   the beginning of the file and used automatically when the db is opened.  You don't need to
///   provide or retrieve it.
struct argon2id_password {
    std::string_view plaintext_pass;

    size_t opslimit = 2;
    size_t memlimit = 19'456 * 1'024;

    // Computes the 32-byte Argon2id hash using `opslimit` and `memlimit` and using the given salt,
    // writing the result into the given output span.
    void compute(std::span<std::byte, 32> out, std::span<const std::byte, 16> salt);

    // Generates a random salt, computes the hash using that salt into `out`, and returns the salt.
    [[nodiscard]] std::array<std::byte, 16> compute(std::span<std::byte, 32> out);
};

/// Specifies a database busy timeout; this handles automatic retry of queries for up to this amount
/// of time in case of an SQLite busy timeout (typically caused by another connection holding a lock
/// on the database).  If not provided or set to a negative duration then the default of 5s is used.
struct busy_timeout {
    std::chrono::milliseconds timeout;

    static constexpr std::chrono::seconds DEFAULT{5};
};

struct wal_mode {
    bool wal = true;
};

/// `no_wal` is a Database constructor value that disables WAL mode.
///
/// Disablying WAL mode is usually a bad idea, but might be needed in exotic circumstances
/// circumstances (e.g. database on a network file system, or on a extremely low memory embedded
/// device).
///
/// If you need to determine this at runtime then instead of `no_wal` pass `wal_mode{use_wal_bool}`.
/// (There is no fixed `use_wal` constant because WAL mode is and will always be the default, and so
/// simply omit it to use WAL mode).
inline constexpr wal_mode no_wal{false};

struct open_readonly {
    bool readonly = true;
};
/// Pass the `readonly` value to the Database constructor to open the database in read-only mode. If
/// omitted then the database is opened in regular read-write mode.  If you need to specify a
/// runtime-dependent value then use `open_readonly{bool}` instead.
inline constexpr open_readonly readonly{true};

struct open_create {
    bool create = true;
};

// Can be used to disallow creation of a database if it does not already exist.  If omitted then a
// new database will be created if the database does not already exist.  `open_create{bool}` is
// available if you need runtime selection.
inline constexpr open_create no_create{false};

class Connection;

// Takes a callback to run immediately after opening the database but before it is returned for use
// by any thread.  This can be used to execute additional initial database pragma or to do things
// like registering user-defined functions.  This only runs the first time the connection is
// established, immediately after opening, *not* on each fetch of the connection from the pool.
//
// Note that the callback could be called from any thread at any time (and so if any required
// synchronization is required it needs to be done inside the callback).
//
// This does nothing if the actual contained function is not set.
struct post_open {
    std::function<void(Connection&)> post_open;
};

namespace detail {
    template <typename T, typename... Types>
    concept any_of = (std::same_as<std::remove_cvref_t<T>, Types> || ...);

    // unique_types<A, B, C> is true if A, B, and C are all unique.
    //
    // Base case for 0 or 1 types:
    template <typename... T>
    constexpr bool unique_types = true;
    // 2+ types:
    template <typename T1, typename T2, typename... Types>
    constexpr bool unique_types<T1, T2, Types...> =
            !std::same_as<T1, T2> && (!std::same_as<T1, Types> && ...) &&
            unique_types<T2, Types...>;

    template <typename T>
    concept MemberObjectPtr = std::is_member_object_pointer_v<T>;

    template <typename T>
    struct member_traits {};
    template <typename V, typename S>
    struct member_traits<V S::*> {
        using class_type = S;
        using value_type = V;
    };

    // Internal class holding a single sqlite connection.
    class conn;

}  // namespace detail

template <typename T>
concept DatabaseEncryptOption = detail::any_of<
        T,
        Encryption,
        plaintext_header_t,
        salt,
        raw_key,
        argon2id_password,
        plaintext_password>;

template <typename T>
concept DatabaseBehaviourOption =
        detail::any_of<T, busy_timeout, wal_mode, open_readonly, open_create, post_open>;

template <typename T>
concept DatabaseOption = DatabaseEncryptOption<T> || DatabaseBehaviourOption<T>;

/// Thrown by `Database::conn()` and `Database::unique_conn()` while the database is suspended (see
/// `Database::suspend()`).  Retriable: the same call succeeds once `Database::resume()` is called.
struct database_suspended : std::runtime_error {
    using std::runtime_error::runtime_error;
};

/// Database connection managing class.  This class operates as a sort of thread-safe SQLite3
/// connection pool where each thread gets its own connection as needed so that multiple threads do
/// not block each other's queries.
///
/// Effectively you should create one single Database instance (typically constructed very early in
/// the application and destroyed very late--the Database must outlive all Connections it spawns),
/// and then use that instance to obtain a database Connection (by calling `.conn()`) whenever you
/// need to use the database.  `conn()` will:
/// - check to see if your thread already has a "checked out" connection, and if so returns a
///   Connection object that shares ownership of the underlying sqlite3 connection with it.
/// - otherwise, if there is a currently available connection in the pool, returns it to you.
/// - otherwise, makes a new connection to the database and returns that.
///
/// The returned Connection object is RAII: when you release it (or when *all* holders of a thread
/// release it, if shared) the Connection object is reinserted into the pool to be available for the
/// next conn() call.
///
/// Each Connection should only be used in a single thread, and is *not thread safe*.  If you need
/// to do things from different threads, simply obtain a Connection from each one.
///
/// The Database connection class always establishes a single connection on construction, to ensure
/// that the database can be successfully opened and decrypted and whatnot.  That connection is
/// immediately added to the connection pool after initial setup.  If you never use a connection
/// from two different threads at the same time, this is the only connection that your application
/// will ever use.
///
/// One note about connections: the Connection object that you obtain by calling `conn()` shared
/// states with any other Connection objects currently obtained by the same thread: i.e. if f() gets
/// a Connection and, while holding that connection (or even in the middle of querying it) calls g()
/// which gets another Connection, both of those connections are the same underlying sqlite
/// connection -- for instance if there is a transaction open in f() then g() is operating in that
/// same transaction.
///
/// If you absolutely want a new connection even if the same thread already has a connection then
/// instead of `conn()` you can call `unique_conn()`: this will give you a connection that is not
/// currently in use, and will not use that connection for any conn() call (until you release it by
/// allowing the Connection object to go out of scope).
///
/// To repeat something said earlier: a single Connection instance must not be used from multiple
/// threads!  There are debug-build assertions to guard against this, but you should be embarrassed
/// if you ever hit one.
///
/// The actual Connection object is non-copyable, but is lightweight and move constructible.
///
/// Each connection also maintains its own cache of re-entrant safe prepared statements which
/// drastically reduce query processing overhead and should be used for most queries (one exception
/// is when a query is known to be run only once, such as initial one-time schema creations/updates.
/// (Re-entrant safe here means that that if you call `SELECT x FROM y WHERE z = ?` from function f,
/// and during processing of this statement you call some other code that happens to invoke the same
/// query for whatever reason, the cache will *not* use the in-use statement [which would be
/// catastrophic] and instead will generate a second cached statement to be used.  That is: each
/// statement can be cached more than once if your code actually needs more than one active at the
/// same time).
class Database {
  private:
    // Various values needed to open a connection (determined during construction):

    // This holds the key or password in the form we need to pass to sqlite_key to set the
    // database key.  That is: plaintext for a normal password, x'...64hex...' for a raw key:
    secure_buffer _key;
    bool _plaintext_header = false;
    std::optional<std::string>
            _pragma_salt;  // Only used if using both plaintext header + plaintext password
    std::filesystem::path _db_path;
    Encryption _enc;
    int _open_flags;
    std::function<void(Connection&)> _post_open;
    std::chrono::milliseconds _busy_timeout;
    bool _wal = true;
    bool _memory = false;

    // Connection containers.  When a thread "checks out" a container, we return from the in_use
    // container if the thread already has a connection.  Otherwise we either remove one from unused
    // (if possible) or make a new one, put it into in_use, and return that.
    //
    // We do a linear scan here for `in_use` because it's unlikely that we will have a large number
    // of threads all wanting sqlite connections, and so linear scan is likely faster than an
    // unordered_map lookup in practice, but also because it means we only have a fixed number of
    // mallocs/frees for a given number of maximum simulaneous DB-using threads, whereas an
    // unordered_map would need malloc/free for every single checkin/checkout, and don't incur
    // unbound growth (e.g. if you are spawning short-lived threads) if we just emptied the
    // shared_ptr but not the node to avoid the free.
    //
    // Unique connections (conn_unique()) go into `_conn_in_use` with a default-constructed thread
    // id (which is guaranteed to never match a real thread id) so that it is still tracked but will
    // never be returned by a thread looking for a match.
    std::vector<std::pair<std::thread::id, std::shared_ptr<detail::conn>>> _conn_in_use;
    std::vector<std::shared_ptr<detail::conn>> _conn_unused;
    int _conn_max_idle = -1;
    Connection get_or_make_conn(std::thread::id tid, int extra_open_flags = 0);
    std::mutex _conn_mutex;

    // Opens and sets up a new connection, outside the pool and without `_post_open`.  Touches no
    // pool state, so needs no lock.
    std::shared_ptr<detail::conn> open_conn(
            int extra_open_flags, std::chrono::milliseconds busy_timeout);

    // Set by `suspend()`, cleared by `resume()`.  `_conn_cv` is notified whenever a connection is
    // returned and on resume: a suspend waiting for held connections waits on it, and so does a
    // held thread waiting for the resume.
    bool _suspended = false;
    std::condition_variable _conn_cv;
    std::function<void()> _on_refused;
    std::vector<std::thread::id> _held;
    std::function<void(std::chrono::milliseconds)> _on_hold;
    std::chrono::milliseconds _hold_interval{std::chrono::seconds{30}};

    // Returns once the database is not suspended: at once if it is not, after the resume on a held
    // thread, and otherwise throws `database_suspended`, after telling `_on_refused`.  Called with
    // `_conn_mutex` held through `lock`.
    void wait_while_suspended(std::unique_lock<std::mutex>& lock);

    // Whether a connection that has just been returned stays open in the idle pool.
    bool keep_idle() const;

    friend class Connection;
    // Called by Connection after releasing a backend connection to have us rescan for any
    // now-unused connections in _conn_in_use.
    void conn_returned();

    // Takes a concrete instance and, if found, copies it into an optional.  If not found returns
    // nullopt.
    template <typename T, typename... Opts>
    static constexpr auto _maybe_instance(Opts&&... opts) {
        using Ret = std::optional<T>;
        if constexpr (sizeof...(Opts) == 0)
            return Ret{std::nullopt};
        else {
            auto finder = []<typename Opt, typename... More>(
                                  auto&& self, Opt&& o, More&&... more) -> Ret {
                if constexpr (std::same_as<std::remove_cvref_t<Opt>, T>)
                    return std::make_optional<T>(std::forward<Opt>(o));
                else if constexpr (sizeof...(More) > 0)
                    return self(self, std::forward<More>(more)...);
                else
                    return std::nullopt;
            };
            return finder(finder, std::forward<Opts>(opts)...);
        }
    }

    // Internal constructor invoked by the templated generic one with all templated options
    // deciphered into positional arguments so that we can keep all the logic in the .cpp rather
    // than the header.
    Database(
            std::filesystem::path db_path,
            std::optional<Encryption> enc,
            std::optional<plaintext_password> plaintext_pass,
            std::optional<raw_key> raw_key,
            std::optional<argon2id_password> argon2id_pass,
            std::optional<plaintext_header_t> plaintext_header,
            std::optional<salt> salt,
            std::optional<busy_timeout> busy_timeout,
            std::optional<wal_mode> wal_mode,
            std::optional<open_create> create,
            std::optional<open_readonly> readonly,
            std::optional<post_open> post_open);

  public:
    // Constructor: this takes a database path followed by any number of optional argument tags (see
    // above) for other supported configuration.  If no encryption value is given but at least one
    // other encryption-related option (e.g. a password or key) is provided, defaults to AEGIS256.
    // If no encryption-related option is provided at all, defaults to no encryption
    // (Encryption::None).
    //
    // The database is created at the given path, creating it if it does not exist.
    //
    // If `db_path` is set to `:memory:` then this opens an ephemeral, in-memory database that is
    // shared by all of this Database's connections (but not with any other Database instance), and
    // is freed once the Database and all of its Connections are destroyed.  This requires SQLite
    // 3.36+ (for the memdb VFS).  Encryption options apply to in-memory databases as they do to
    // files: the in-memory database pages are stored encrypted (though, as with any encrypted
    // database, pages in the connection's page cache are not).  In-memory databases never use WAL
    // mode (wal_mode is ignored), and you must not enable it yourself (e.g. from post_open): on an
    // encrypted in-memory database that segfaults due to a sqlite3mc bug.
    template <DatabaseOption... Opt>
    Database(std::filesystem::path db_path, const Opt&... opts) :
            Database{
                    std::move(db_path),
                    (DatabaseEncryptOption<Opt> || ...)
                            ? _maybe_instance<Encryption>(opts...)
                            : std::optional<Encryption>{Encryption::None},
                    _maybe_instance<plaintext_password>(opts...),
                    _maybe_instance<raw_key>(opts...),
                    _maybe_instance<argon2id_password>(opts...),
                    _maybe_instance<plaintext_header_t>(opts...),
                    _maybe_instance<salt>(opts...),
                    _maybe_instance<busy_timeout>(opts...),
                    _maybe_instance<wal_mode>(opts...),
                    _maybe_instance<open_create>(opts...),
                    _maybe_instance<open_readonly>(opts...),
                    _maybe_instance<post_open>(opts...)} {
        static_assert(
                detail::unique_types<Opt...>,
                "Database constructor option arguments must be unique");
        static_assert(
                ((detail::any_of<Opt, plaintext_password, raw_key, argon2id_password> ? 1 : 0) +
                 ... + 0) <= 1,
                "Database: plaintext_password/raw_key/argon2id_password constructor options are "
                "mutually exclusive");
    }

    // Sets a maximum number of idle connections.  If this has been called then whenever a
    // Connection object is released and returned to the pool, if the free connection pool would
    // exceed this size the connection is closed instead.  If set too low this can result in a lot
    // of database closing and re-opening.  If you set it to 0 then you basically force every single
    // non-recursive call to `conn()` to open a new database and close it when destructed.
    //
    // If this has not been called, or this is called with a negative value, then there will be no
    // current limit: the number of idle connections will simply be however many have ever been
    // needed at once (and are not currently used).
    //
    // If there are already more than this number of idle connections then this closes the extras
    // so that the given limit now applies.
    void set_max_idle_conns(int max_idle);

    // What `suspend()` managed within its wait.
    struct suspend_result {
        // Every connection was closed, so the process holds no lock on the database files.  False
        // when a held connection was not returned in time, or `resume()` was called meanwhile.
        bool closed = false;
        // The WAL was checkpointed and truncated.  Only attempted when asked for, and only once
        // every connection was closed.
        bool checkpointed = false;
    };

    // Stops the database being used until `resume()`, and closes every connection so the process
    // holds no lock on its files: iOS kills a suspended app that holds a lock on a file in a shared
    // container.
    //
    // From this call on, `conn()` and `unique_conn()` throw `database_suspended` -- except to a
    // thread already holding a connection, which keeps it so that it can finish what it started,
    // and to a thread registered with `hold_while_suspended()`, which waits for `resume()`.
    // Idle connections close at once, and this waits up to `wait` for the held ones to be returned,
    // each closing as it is.  With `interrupt`, for when there is no time left to wait, the
    // statements running on held connections are interrupted first: an interrupted statement
    // throws on its own thread, where the transaction's RAII rollback runs.
    //
    // With `checkpoint`, once every connection is closed one is opened just long enough to
    // checkpoint and truncate the WAL, within what remains of `wait`.  Closing the last connection
    // to a database does that by itself; this is for when another process -- an app's extension --
    // still has the database open, so that this process closing is not the last.
    //
    // An in-memory database keeps one idle connection open, since closing the last would discard
    // it; having no files, it holds no lock either way.
    suspend_result suspend(
            std::chrono::milliseconds wait, bool interrupt = false, bool checkpoint = false);

    // Allows connections again after `suspend()`.  They reopen on the next `conn()`, not here.
    void resume();

    // Whether the database is suspended.
    bool suspended();

    // Called, on the refused thread, each time `conn()` or `unique_conn()` refuses a connection
    // because the database is suspended, just before throwing: work while suspended is a bug in
    // whatever forgot to resume, and this is where to make it loud.  Called with the pool's lock
    // held, so it must not touch the database.
    void on_refused(std::function<void()> f);

    // Makes `conn()` and `unique_conn()` on thread `tid` wait while the database is suspended,
    // until `resume()`, rather than throw.  For a thread whose work cannot all be stopped at a
    // suspend -- an event loop with jobs already queued, or answers still to arrive -- and is
    // better paused than failed: the job carries on once resumed, or never finishes if the process
    // is killed meanwhile, which a suspend has to allow for anyway.  A held thread waits with no
    // limit, so the database must be resumed before it is destroyed.
    void hold_while_suspended(std::thread::id tid);

    // Undoes `hold_while_suspended()`, for a thread that is ending: its id may be reused.
    void stop_holding(std::thread::id tid);

    // Called on a held thread when it begins waiting for a resume, with 0, and again every
    // `interval` that it is still waiting, with how long it has waited: a short wait is expected
    // work arriving while suspended, a long one a resume that never came.  Called with the pool's
    // lock held, so it must not touch the database.
    void on_hold(
            std::function<void(std::chrono::milliseconds waited)> f,
            std::chrono::milliseconds interval = std::chrono::seconds{30});

    // How many connections are open, idle and in use together: what a suspend that closed every
    // connection leaves at zero (one, for an in-memory database).
    size_t open_connections();

    // Gets a connection.  If the current thread already has a connection checked out, this returns
    // a Connection object that shared connection ownership with the other connection.
    //
    // This is typically fine: the same connection can perfectly well handle multiple simultaneous
    // queries.
    //
    // One case where it can be problematic, however, is if you are modifying tables that you are in
    // the middle of querying: e.g. for example if function f() is in the middle of iterating
    // through a result set and calls g() which executes a deletion or update that affects rows in
    // that result set.  Generally avoid that by fetching results before going off into other
    // functions that modify the same rows, and if that is really infeasible, get a separate
    // connection (which gives you atomicity through the WAL).
    //
    // If all available connections are in use (and this thread does not currently have one) this
    // makes a new one.
    Connection conn();

    // Gets a connection that is not (and will not be) shared with other threads from the pool.
    // Makes a new connection if there are no currently free threads.  Note that once you are
    // finished with the connection it will be returned to the pool and so might be shared later,
    // and might have been shared in the past; this just ensures that it won't be shared while you
    // hold the Connection instance.
    Connection unique_conn();

    // Destructor.  This object should be destroyed while any Connections produced by this object
    // still remain.
    ~Database();
};

/// Wrapper around a SQLite::Statement that calls `tryReset()` on destruction of the wrapper. This
/// is not typically invoked directly but rather is produced by calling prepared_st() or
/// prepared_bind().
class StatementWrapper {
    detail::conn& conn;
    std::unique_ptr<SQLite::Statement> st;

    friend class detail::conn;
    StatementWrapper(detail::conn& conn, std::unique_ptr<SQLite::Statement> st) noexcept :
            conn{conn}, st{std::move(st)} {}

  public:
    // Move constructible, but not move assignable or copyable.
    StatementWrapper(StatementWrapper&&) = default;
    StatementWrapper& operator=(StatementWrapper&&) = delete;
    StatementWrapper(const StatementWrapper&) = delete;
    StatementWrapper& operator=(const StatementWrapper&) = delete;

    ~StatementWrapper();

    explicit operator bool() const { return (bool)st; }

    SQLite::Statement& operator*() noexcept {
        assert(st);
        return *st;
    }
    const SQLite::Statement& operator*() const noexcept {
        assert(st);
        return *st;
    }
    SQLite::Statement* operator->() noexcept {
        assert(st);
        return st.get();
    }
    const SQLite::Statement* operator->() const noexcept {
        assert(st);
        return st.get();
    }
    operator SQLite::Statement&() noexcept {
        assert(st);
        return *st;
    }
    operator const SQLite::Statement&() const noexcept {
        assert(st);
        return *st;
    }
};

/** Extends the above with the ability to iterate through results. */
template <typename... T>
class IterableStatementWrapper : StatementWrapper {
  public:
    using StatementWrapper::StatementWrapper;

    IterableStatementWrapper(StatementWrapper&& s) : StatementWrapper{std::move(s)} {}

    class iterator {
        IterableStatementWrapper& sw;
        bool finished;
        explicit iterator(IterableStatementWrapper& sw, bool finished = false) :
                sw{sw}, finished{finished} {
            ++*this;
        }
        friend class IterableStatementWrapper;

      public:
        iterator(const iterator&) = delete;
        iterator(iterator&&) = delete;
        void operator=(const iterator&) = delete;
        void operator=(iterator&&) = delete;

        detail::type_or_tuple<T...> operator*() { return get<T...>(sw); }

        iterator& operator++() {
            if (!finished)
                finished = !sw->executeStep();
            return *this;
        }
        void operator++(int) { ++*this; }
        bool operator==(const iterator& other) {
            return &sw == &other.sw && finished == other.finished;
        }
        bool operator!=(const iterator& other) { return !(*this == other); }

        using value_type = detail::type_or_tuple<T...>;
        using reference = value_type;
        using difference_type = std::ptrdiff_t;
        using pointer = const value_type*;
        using iterator_category = std::input_iterator_tag;
    };

    iterator begin() { return iterator{*this}; }
    iterator end() { return iterator{*this, true}; }
};

// This is the per-thread shared connection class returned by Database::conn().  While held this
// connection is reserved for the single thread to which it was requested, and must not be used in
// any other thread.
class Connection {
    std::shared_ptr<detail::conn> _conn;

    friend class Database;
    Connection(Database& db, std::shared_ptr<detail::conn> conn);

  public:
    // Movable so that Database code holding _conn_mutex can return one by value; see ~Connection.
    // Not copyable: conn_returned() reads use_count() == 1 as "nothing is using this connection".
    Connection(Connection&& c) noexcept : _conn{std::move(c._conn)}, db{c.db}, sql{c.sql} {}
    Connection(const Connection&) = delete;
    Connection& operator=(const Connection&) = delete;
    Connection& operator=(Connection&&) = delete;

    // The owning Database
    Database& db;

    // The SQLiteCpp instance to actually interact with the sqlite3 db.
    SQLite::Database& sql;

    /// Prepares a query, caching it, and returns a wrapper that automatically resets the
    /// prepared statement on destruction.
    StatementWrapper prepared_st(std::string_view query);

    /// Prepares (with caching) and binds a query, returning the active statement handle.  Like
    /// `prepared_st` the wrapper resets the prepared statement on destruction.
    template <typename... T>
    StatementWrapper prepared_bind(std::string_view query, const T&... bind) {
        auto st = prepared_st(query);
        bind_oneshot(st, bind...);
        return st;
    }

    /// Prepares (with caching), binds parameters, then returns an object that lets you iterate
    /// through results where each row is a T or tuple<T...>:
    template <typename... T, typename... Bind>
        requires(sizeof...(T) != 0)
    IterableStatementWrapper<T...> prepared_results(std::string_view query, const Bind&... bind) {
        return IterableStatementWrapper<T...>{prepared_bind(query, bind...)};
    }

    /// Prepares (with caching) a query that returns no rows and then executes it, optionally
    /// binding the given parameters when executing.  Throws if the query fails or returns any rows.
    template <typename... T>
    int prepared_exec(std::string_view query, const T&... bind) {
        return exec_query(prepared_st(query), bind...);
    }

    /// Prepares (with caching) a query that returns a single row (with optional bind
    /// parameters), executes it, and returns the value.  Throws if the query returns 0 or more
    /// than 1 rows.
    template <typename... T, typename... Bind>
        requires(!(holds_blob_view<Bind> || ...))
    auto prepared_get(std::string_view query, const Bind&... bind) {
        return exec_and_get<T...>(prepared_st(query), bind...);
    }

    /// Prepares (with caching) a query that returns at most a single row (with optional bind
    /// parameters), executes it, and returns the value or nullopt if the query returned no
    /// rows. Throws if the query returns more than 1 rows.
    template <typename... T, typename... Bind>
        requires(!(holds_blob_view<Bind> || ...))
    auto prepared_maybe_get(std::string_view query, const Bind&... bind) {
        return exec_and_maybe_get<T...>(prepared_st(query), bind...);
    }

    // Queries whether a table, index, or table of the given name exists.
    bool table_exists(std::string_view table_name);
    bool index_exists(std::string_view index_name);
    bool trigger_exists(std::string_view trigger_name);

    struct ColumnInfo {
        std::string name;  ///< The column name
        std::string type;  ///< The type (as given at table creation time, *not* normalized)
        bool not_null;     ///< True if the column is non-nullable, false if nullable
        bool has_default;  ///< True if the column has a non-null default, false otherwise
        int primary_key;   ///< 0 for regular columns; otherwise 1-N for the N columns in the PK
    };

    // Queries and returns a table's columns; typically used for database creation/upgrading.
    std::vector<ColumnInfo> get_columns(std::string_view table_name);

    // Destroying this connection object drops its lease on the underlying connection and,
    // if this is the last lease, the connection is returned to the owner Database's connection
    // pool.
    //
    // Acquires _conn_mutex to do so: a Connection must not be allowed to destruct anywhere in
    // Database that already holds that lock, or the thread deadlocks against itself.
    ~Connection();
};

}  // namespace session::sqlite
