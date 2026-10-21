#include <thread>
#ifdef SESSION_SQLITE_SQLCIPHER
// sqlcipher for reasons that haven't been applicable for a decade doesn't expose its key API unless
// this is defined, so we need to ensure this is defined before anything that might include
// sqlite.hpp:
#define SQLITE_HAS_CODEC
#endif

#include <SQLiteCpp/Database.h>
#include <sodium/core.h>
#include <sodium/crypto_pwhash.h>
#include <sodium/randombytes.h>
#include <sodium/utils.h>
#include <sqlite3.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <fstream>
#include <mutex>
#include <session/sqlite.hpp>
#include <stdexcept>

#include "conn.hpp"

namespace session::sqlite {

static void sodium_initialize() {
    if (sodium_init() == -1)
        throw std::runtime_error{"Sodium init failed"};
}

// Writes the lowercase hex representation of `bytes` to `out`, which must have room for
// 2*bytes.size() characters.
static void to_hex(std::span<const std::byte> bytes, char* out) {
    constexpr char digits[] = "0123456789abcdef";
    for (auto b : bytes) {
        auto v = std::to_integer<unsigned char>(b);
        *out++ = digits[v >> 4];
        *out++ = digits[v & 0xf];
    }
}

// Returns the lowercase hex representation of `bytes` as a string.
static std::string to_hex(std::span<const std::byte> bytes) {
    std::string result(bytes.size() * 2, '\0');
    to_hex(bytes, result.data());
    return result;
}

using namespace std::literals;

static std::atomic<uint64_t> memdb_counter{0};

bool enabled(Encryption type) {
    switch (type) {
        case Encryption::None: return true;
        case Encryption::ChaCha20:
        case Encryption::AEGIS256:
        case Encryption::Ascon128:
#ifdef SESSION_SQLITE_MULTIPLE_CIPHERS
            return true;
#else
            return false;
#endif
        case Encryption::SQLCipher3:
        case Encryption::SQLCipher4:
#if defined(SESSION_SQLITE_MULTIPLE_CIPHERS) || defined(SESSION_SQLITE_SQLCIPHER)
            return true;
#else
            return false;
#endif
    };
    return false;
}

Database::Database(
        std::filesystem::path db_path,
        std::optional<Encryption> enc,
        std::optional<plaintext_password> plaintext_pass,
        std::optional<raw_key> raw_key,
        std::optional<argon2id_password> argon2id_pass,
        std::optional<plaintext_header_t> plain_hdr,
        std::optional<salt> pw_salt,
        std::optional<busy_timeout> busy_t_o,
        std::optional<wal_mode> wal_mode,
        std::optional<open_create> create,
        std::optional<open_readonly> readonly,
        std::optional<post_open> post_open) :
        // enc is nullopt only when encryption-related options were given but Encryption itself
        // was not; in that case AEGIS256 is the default.  When no encryption-related options are
        // given at all, the public constructor passes Encryption::None explicitly.
        _db_path{std::move(db_path)}, _enc{enc.value_or(Encryption::AEGIS256)} {

    if (!enabled(_enc))
        throw std::runtime_error{
                "Database: this build does not support the selected encryption type"};
    if (_enc == Encryption::None) {
        if (raw_key || plaintext_pass || argon2id_pass || plain_hdr || pw_salt)
            throw std::invalid_argument{
                    "Database: cannot use Encryption::None with database encryption options"};
    } else {
        if (!(raw_key || plaintext_pass || argon2id_pass))
            throw std::invalid_argument{
                    "Database: opening an encrypted database requires one of "
                    "raw_key/plaintext_password/argon2id_password"};

        if (plain_hdr && !pw_salt) {
            if (!raw_key)
                throw std::invalid_argument{
                        "Database: opening an encrypted database with a password in plaintext "
                        "header mode requires a salt argument"};

            if ((_enc == Encryption::SQLCipher3 || _enc == Encryption::SQLCipher4))
                throw std::invalid_argument{
                        "Database: opening an sqlcipher-encrypted file in plaintext header mode "
                        "requires a "
                        "salt argument (even in raw_key mode)"};
        }
    }

    // If you have *explicitly* given create=true and readonly=true then that is a
    // error.  If you've only given readonly but omitted create then we imply create is
    // false.
    if (create && create->create && readonly && readonly->readonly)
        throw std::invalid_argument{
                "Database: conflict constructor arguments: readonly mode cannot be combined with "
                "an explicit open_create{true} argument"};

    _open_flags = (readonly && readonly->readonly ? SQLite::OPEN_READONLY : SQLite::OPEN_READWRITE)
                // "NOMUTEX" here actually means no *full* mutex, but makes sqlite safe to use from
                // multiple threads as long as a connection (and any connection-derived objects) are
                // not used from multiple threads.
                | SQLite::OPEN_NOMUTEX;
    _busy_timeout =
            (busy_t_o && busy_t_o->timeout >= 0s) ? busy_t_o->timeout : busy_timeout::DEFAULT;
    _wal = wal_mode ? wal_mode->wal : true;

    // A plain ":memory:" gives every connection its own private database, so pooled connections
    // wouldn't see each other.  A memdb VFS name starting with "/" is shared by all connections
    // that open it, and the counter keeps separate Database instances apart (even one constructed
    // where a destroyed one used to be).
    if (_db_path == ":memory:") {
        // sqlite3mc only wraps the default VFS with its encryption shim, so naming plain memdb would
        // bypass encryption entirely; asking for "multipleciphers-memdb" makes it create a shim
        // around memdb on demand.
        const char* vfs = "memdb";
#ifdef SESSION_SQLITE_MULTIPLE_CIPHERS
        if (_enc != Encryption::None)
            vfs = "multipleciphers-memdb";
#endif
        _db_path = "file:/session-sqlite-memdb-" + std::to_string(++memdb_counter) +
                   "?vfs=" + vfs;
        _open_flags |= SQLite::OPEN_URI;
        _memory = true;

        // memdb has no shared memory support, so WAL is impossible.  Plain memdb just refuses the
        // journal_mode change, but sqlite3mc's shim advertises an xShmMap that forwards to memdb's
        // NULL one, so enabling WAL through it segfaults on the first read.
        _wal = false;
    }

    std::optional<std::array<std::byte, 16>> salt;

    if (plain_hdr) {
        _plaintext_header = true;
        if (_enc == Encryption::SQLCipher3)
            throw std::invalid_argument{"SQLCipher3 and plaintext_header are incompatible"};
        if (pw_salt) {
            salt.emplace();
            std::memcpy(salt->data(), pw_salt->salt.data(), pw_salt->salt.size());
        }
    }

    auto store_raw_key = [&](std::span<const std::byte, 32> key) {
        // We have to pass this to sqlcipher or sqlite-mc as "x'...'" where ... is hex; sqlite-mc
        // would allow us to pass bytes via `raw:BYTES` but we just use the x'...' there too to only
        // have one codepath.
        //
        // In theory we can include the salt here on the end, but that hits sqlite3-mc issue #226,
        // so we send the salt via PRAGMA instead to work around that.
        auto rw = _key.resize(67);
        rw.buf[0] = std::byte{'x'};
        rw.buf[1] = std::byte{'\''};
        to_hex(key, reinterpret_cast<char*>(rw.buf.data() + 2));
        rw.buf.back() = std::byte{'\''};
    };

    if (argon2id_pass) {
        bool salt_from_file = false;
        if (!salt) {
            salt.emplace();
            if (exists(_db_path)) {
                // If we aren't using a plaintext header then the first 16 bytes of the file are the
                // salt.  If the file doesn't exist, we select a random one and specify that when
                // opening the file.
                std::ifstream f;
                f.exceptions(std::ifstream::failbit | std::ifstream::badbit);
                f.open(_db_path, std::ios::binary);
                f.read(reinterpret_cast<char*>(salt->data()), salt->size());

                if (std::string_view{reinterpret_cast<const char*>(salt->data()), salt->size()} ==
                    "SQLite Format 3\0"sv)
                    throw std::runtime_error{
                            "Database exists but uses a plaintext header (or is not encrypted); "
                            "plaintext header mode is required (using plaintext_header_salt)"};

                salt_from_file = true;
            } else {
                randombytes_buf(salt->data(), salt->size());
            }
        }

        std::array<std::byte, 32> key;
        argon2id_pass->compute(key, *salt);

        if (salt_from_file)
            salt.reset();  // We don't need to pass it: sqlite will read it itself

        store_raw_key(key);

        sodium_memzero(key.data(), key.size());

    } else if (raw_key) {

        store_raw_key(raw_key->key);

    } else if (plaintext_pass) {
        // TODO FIXME: we could do a lot better here by precomputing the PBKDF2-SHA512-HMAC (or
        // whatever) and storing that as a key so that we don't make the database layer do the
        // PBKDF2 on every new connection.
        //
        // But really you should just switch to argon2 instead because PBKDF2 doesn't offer much
        // practical modern protection.

        _key.update(std::span{
                reinterpret_cast<const std::byte*>(plaintext_pass->pass.data()),
                plaintext_pass->pass.size()});
    }

    if (salt)
        _pragma_salt.emplace(
                "PRAGMA cipher_salt = '" + to_hex(*salt) + "'");

    if (post_open && post_open->post_open)
        _post_open = std::move(post_open->post_open);

    // Get an initial connection so that we are testing that we can connect here in the constructor.
    // We immediately drop it, which returns that single connection to the idle conns pool to be
    // returned as soon as someone calls `conn()` to get it (without needing to reconnect again).

    // std::lock_guard lock{_conn_mutex}; // Not necessary - we're still in the constructor
    get_or_make_conn(std::thread::id{}, !create || create->create ? SQLite::OPEN_CREATE : 0);
}

Connection::Connection(Database& db, std::shared_ptr<detail::conn> conn) :
        db{db}, _conn{std::move(conn)}, sql{_conn->sql} {}

Connection::~Connection() {
    if (_conn) {
        _conn.reset();
        db.conn_returned();
    }
}

void Database::conn_returned() {
    // Called during Connection destruction after releasing the detail::conn shared_ptr to return
    // any now-unused connections from _conn_in_use into _conn_unused.
    std::lock_guard lock{_conn_mutex};
    for (auto it = _conn_in_use.begin(); it != _conn_in_use.end();) {
        auto& [th, conn] = *it;
        assert(conn);  // If we end up with a dead pointer in here then some code is broken
        if (conn.use_count() == 1) {
            if (keep_idle())
                _conn_unused.push_back(std::move(conn));
            // else we have enough idle connections (or are suspended) so let it drop

            it = _conn_in_use.erase(it);
        } else
            ++it;
    }
    _conn_cv.notify_all();
}

bool Database::keep_idle() const {
    if (_suspended)
        return _memory && _conn_unused.empty();
    return _conn_max_idle < 0 || _conn_unused.size() < static_cast<size_t>(_conn_max_idle);
}

Database::suspend_result Database::suspend(
        std::chrono::milliseconds wait, bool interrupt, bool checkpoint) {
    auto deadline = std::chrono::steady_clock::now() + wait;
    std::unique_lock lock{_conn_mutex};
    _suspended = true;

    if (_conn_unused.size() > (_memory ? 1 : 0))
        _conn_unused.resize(_memory ? 1 : 0);
    if (interrupt)
        for (auto& [th, c] : _conn_in_use)
            sqlite3_interrupt(c->sql.getHandle());

    suspend_result result;
    result.closed = _conn_cv.wait_until(lock, deadline, [this] {
        return !_suspended || _conn_in_use.empty();
    }) && _suspended;

    if (!result.closed || !checkpoint || _memory || !_wal)
        return result;

    // A connection of its own, outside the pool and opened without its lock, so that a resume can
    // go ahead meanwhile and never sees it.  Opening reads the database, so it too is held to what
    // remains of `wait`.
    lock.unlock();
    try {
        auto left = std::max(
                std::chrono::milliseconds{0},
                std::chrono::duration_cast<std::chrono::milliseconds>(
                        deadline - std::chrono::steady_clock::now()));
        auto c = open_conn(0, left);
        // One row: whether it was blocked, the WAL's frames, and how many were checkpointed.
        SQLite::Statement st{c->sql, "PRAGMA wal_checkpoint(TRUNCATE)"};
        result.checkpointed = st.executeStep() && st.getColumn(0).getInt() == 0;
    } catch (const std::exception&) {
        // Busy past the deadline, or the WAL could not be truncated: the WAL is left as it is.
    }
    return result;
}

void Database::resume() {
    std::lock_guard lock{_conn_mutex};
    _suspended = false;
    _conn_cv.notify_all();
}

bool Database::suspended() {
    std::lock_guard lock{_conn_mutex};
    return _suspended;
}

void Database::on_refused(std::function<void()> f) {
    std::lock_guard lock{_conn_mutex};
    _on_refused = std::move(f);
}

void Database::hold_while_suspended(std::thread::id tid) {
    std::lock_guard lock{_conn_mutex};
    if (std::find(_held.begin(), _held.end(), tid) == _held.end())
        _held.push_back(tid);
}

void Database::stop_holding(std::thread::id tid) {
    std::lock_guard lock{_conn_mutex};
    std::erase(_held, tid);
}

void Database::on_hold(
        std::function<void(std::chrono::milliseconds)> f, std::chrono::milliseconds interval) {
    std::lock_guard lock{_conn_mutex};
    _on_hold = std::move(f);
    _hold_interval = interval;
}

size_t Database::open_connections() {
    std::lock_guard lock{_conn_mutex};
    return _conn_in_use.size() + _conn_unused.size();
}

void Database::wait_while_suspended(std::unique_lock<std::mutex>& lock) {
    if (!_suspended)
        return;
    if (std::find(_held.begin(), _held.end(), std::this_thread::get_id()) == _held.end()) {
        if (_on_refused)
            _on_refused();
        throw database_suspended{"the database is suspended"};
    }
    auto start = std::chrono::steady_clock::now();
    for (auto next = start; _suspended;) {
        if (auto now = std::chrono::steady_clock::now(); now >= next) {
            if (_on_hold)
                _on_hold(std::chrono::duration_cast<std::chrono::milliseconds>(now - start));
            next = now + _hold_interval;
        }
        _conn_cv.wait_until(lock, next, [this] { return !_suspended; });
    }
}

void Database::set_max_idle_conns(int max_idle) {
    std::lock_guard lock{_conn_mutex};

    _conn_max_idle = max_idle;
    if (_conn_max_idle >= 0 && _conn_unused.size() > static_cast<size_t>(_conn_max_idle))
        _conn_unused.resize(_conn_max_idle);
}

Connection Database::get_or_make_conn(std::thread::id tid, int extra_open_flags) {

    if (!_conn_unused.empty()) {
        auto c = std::move(_conn_unused.back());
        _conn_unused.pop_back();
        _conn_in_use.emplace_back(tid, c);
        c->reset_thread();
        return Connection{*this, std::move(c)};
    }

    // Run out of idle connections so we need to make a new one (this also gets called during
    // construction for the very first connection)
    auto conn = open_conn(extra_open_flags, _busy_timeout);

    Connection c{*this, conn};

    if (_post_open)
        _post_open(c);

    _conn_in_use.emplace_back(tid, std::move(conn));

    return c;
}

std::shared_ptr<detail::conn> Database::open_conn(
        int extra_open_flags, std::chrono::milliseconds busy_timeout) {
    auto conn = std::make_shared<detail::conn>(
            _db_path, _open_flags | extra_open_flags, static_cast<int>(busy_timeout.count()));

    auto& sql = conn->sql;

    auto connect_pragma = [&](const std::string& pragma, const char* failing_thing) {
        if (char* errmsg;
            SQLITE_OK != sqlite3_exec(sql.getHandle(), pragma.c_str(), nullptr, nullptr, &errmsg)) {
            auto err = "Failed to "s + failing_thing + ": "s + errmsg;
            sqlite3_free(errmsg);
            throw std::runtime_error{std::move(err)};
        }
    };

    connect_pragma(
            "PRAGMA trusted_schema = OFF", "disable horrible default trusted_schema setting");

#ifdef SESSION_SQLITE_SQLCIPHER
    // SQLCipher
    if (_enc == Encryption::SQLCipher3)
        connect_pragma("PRAGMA cipher_compatibility = 3", "enable sqlcipher3 compat mode");
    else if (_enc != Encryption::SQLCipher4 & _enc != Encryption::None)
        throw std::runtime_error{
                "The given encryption type is not supported when using the SQLCipher backend"};

#elif defined(SESSION_SQLITE_MULTIPLE_CIPHERS)
    // SQLite3 multiple ciphers
    const char* name = nullptr;
    int legacy = -1;
    int algorithm = -1;
    switch (_enc) {
        case Encryption::AEGIS256:
            name = "aegis";
            algorithm = 4;  // Magic sqlite3-mc constant for AEGIS-256
            break;
        case Encryption::Ascon128: name = "ascon128"; break;
        case Encryption::ChaCha20: name = "chacha20"; break;
        case Encryption::SQLCipher3:
            name = "sqlcipher";
            legacy = 3;
            break;
        case Encryption::SQLCipher4:
            name = "sqlcipher";
            legacy = 4;
            break;
        case Encryption::None: break;
    };

    if (name) {
        int cipher_idx = sqlite3mc_cipher_index(name);
        if (cipher_idx == -1)
            throw std::runtime_error{"Current SQLite-MC build does not support cipher: "s + name};
        if (-1 == sqlite3mc_config(sql.getHandle(), "cipher", cipher_idx))
            throw std::runtime_error{"Cipher selection failed"};
        if (legacy >= 0)
            if (-1 == sqlite3mc_config_cipher(sql.getHandle(), name, "legacy", legacy))
                throw std::runtime_error{"SQLCipher version mode selection failed"};
        if (algorithm >= 0)
            if (-1 == sqlite3mc_config_cipher(sql.getHandle(), name, "algorithm", algorithm))
                throw std::runtime_error{"Failed to set cipher algorithm"};
    }
#else
    // A stock sqlite3, with no encryption support of any kind compiled in.
    if (_enc != Encryption::None)
        throw std::runtime_error{
                "Encryption was requested, but this session-sqlite was built against a plain "
                "sqlite3 with no encryption support; rebuild against SQLite3-MC or SQLCipher"};
#endif

    if (_plaintext_header)
        connect_pragma(
                _enc == Encryption::SQLCipher4 ? "PRAGMA plaintext_header_size = 32"
                                               : "PRAGMA plaintext_header_size = 24",
                "enable plaintext header mode");

    if (_pragma_salt)
        connect_pragma(*_pragma_salt, "set cipher salt");

#if defined(SESSION_SQLITE_SQLCIPHER) || defined(SESSION_SQLITE_MULTIPLE_CIPHERS)
    if (_enc != Encryption::None) {
        auto ro = _key.access();
        // A failure here (e.g. a VFS without the encryption shim) would otherwise carry on and
        // silently give an unencrypted database.
        if (int rc = sqlite3_key(sql.getHandle(), ro.buf.data(), ro.buf.size()); rc != SQLITE_OK)
            throw std::runtime_error{
                    "Failed to set database encryption key: "s + sqlite3_errstr(rc)};
    }
#endif

    // Now make sure we can query something: this is our failure point (via exception) if the
    // authentication key is incorrect as this will be the first place that an actual read happens.
    sql.exec("SELECT COUNT(*) FROM sqlite_master");

    if (_wal)
        // Don't fail on these because we can still work (albeit slower) even if they fail for some
        // reason
        if (sql.tryExec("PRAGMA journal_mode = WAL") == SQLite::OK)
            sql.tryExec("PRAGMA synchronous = NORMAL");

    // Foreign key enforcement being off is another absolutely miserable sqlite3 default, but
    // apparently we not only have to set it but also verify that it has been set because some
    // versions and or compilation options can just silently *not* set it because it's better to
    // have horrible defaults and silently ignore fixing them rather than risk breaking some
    // sqlite-using PHP script from the late 90s.
    try {
        sql.exec("PRAGMA foreign_keys = ON");
        int fk_enabled = sql.execAndGet("PRAGMA foreign_keys").getInt();
        if (fk_enabled != 1)
            throw std::runtime_error{
                    "Foreign key enabling query succeeded, but foreign keys were not actually "
                    "enabled!"};
    } catch (const std::exception& e) {
        throw std::runtime_error{"Failed to enable foreign key integrity: "s + e.what()};
    }

    return conn;
}

Connection Database::unique_conn() {
    std::unique_lock lock{_conn_mutex};
    wait_while_suspended(lock);
    return get_or_make_conn(std::thread::id{});
}

Connection Database::conn() {
    std::unique_lock lock{_conn_mutex};

    auto t = std::this_thread::get_id();
    for (auto& [th, conn] : _conn_in_use)
        if (th == t)
            return Connection{*this, conn};

    wait_while_suspended(lock);
    return get_or_make_conn(t);
}

Database::~Database() {
    assert(_conn_in_use.empty());
}

StatementWrapper Connection::prepared_st(std::string_view query) {
    return _conn->prepared_st(query);
}

static const auto item_exists =
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type = ? AND name = ?)"s;
bool Connection::table_exists(std::string_view table_name) {
    return prepared_get<int>(item_exists, "table"sv, table_name);
}
bool Connection::index_exists(std::string_view index_name) {
    return prepared_get<int>(item_exists, "index"sv, index_name);
}
bool Connection::trigger_exists(std::string_view trigger_name) {
    return prepared_get<int>(item_exists, "trigger"sv, trigger_name);
}

std::vector<Connection::ColumnInfo> Connection::get_columns(std::string_view table_name) {
    std::vector<ColumnInfo> cols;
    for (auto [name, type, notnull, has_default, pk] :
         prepared_results<std::string, std::string, int, int, int>(
                 "SELECT name, type, \"notnull\", dflt_value is not null, pk"
                 " FROM pragma_table_info(?)",
                 table_name)) {
        auto& col = cols.emplace_back();
        col.name = std::move(name);
        col.type = std::move(type);
        col.not_null = notnull;
        col.has_default = has_default;
        col.primary_key = pk;
    }
    return cols;
}

StatementWrapper::~StatementWrapper() {
    if (st) {
        st->tryReset();
        conn.statement_finished(std::move(st));
    }
}

void argon2id_password::compute(std::span<std::byte, 32> out, std::span<const std::byte, 16> salt) {
    sodium_initialize();

    if (0 != crypto_pwhash(
                     reinterpret_cast<unsigned char*>(out.data()),
                     out.size(),

                     plaintext_pass.data(),
                     plaintext_pass.size(),
                     reinterpret_cast<const unsigned char*>(salt.data()),
                     opslimit,
                     memlimit,
                     crypto_pwhash_argon2id_ALG_ARGON2ID13))
        throw std::runtime_error{"Failed to compute argon2 hash (most likely insufficient memory)"};
}

std::array<std::byte, 16> argon2id_password::compute(std::span<std::byte, 32> out) {
    sodium_initialize();
    std::array<std::byte, 16> salt;
    randombytes_buf(salt.data(), salt.size());
    compute(out, salt);
    return salt;
}

}  // namespace session::sqlite
