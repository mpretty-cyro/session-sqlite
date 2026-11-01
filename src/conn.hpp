#pragma once

#include <SQLiteCpp/Database.h>
#include <SQLiteCpp/Statement.h>

#include <functional>
#include <memory>
#include <session/sqlite.hpp>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

namespace session::sqlite::detail {

// Transparent hash so that the statement cache can be searched by string_view without constructing
// a std::string key for every lookup.
struct string_hash {
    using is_transparent = void;
    size_t operator()(std::string_view s) const noexcept { return std::hash<std::string_view>{}(s); }
};

// Internal wrapper that holds a single connection and a statement cache associated with that
// connection.
class conn {
  public:
    SQLite::Database sql;

    template <typename... Args>
    conn(Args&&... args) : sql{std::forward<Args>(args)...} {}

    void reset_thread() { _thread = std::this_thread::get_id(); }

    StatementWrapper prepared_st(std::string_view query);

    void statement_finished(std::unique_ptr<SQLite::Statement> st);

    // Replaces sqlite's automatic checkpoint, which runs inside the COMMIT that takes the WAL past
    // the connection's wal_autocheckpoint threshold, with one at that same threshold that, for a
    // COMMIT in an ImmediateTransaction, waits until the transaction has released its Database's
    // write lock, so that the writers queued behind it don't wait for the checkpoint too.
    //
    // Reads the threshold when called, so whatever sets it must run first; setting it afterwards
    // reinstalls sqlite's own hook in place of this one.
    void install_wal_hook();

    // Set by an ImmediateTransaction while it holds the write lock with this connection.
    bool in_immediate = false;
    // Set when a COMMIT in that transaction has left a checkpoint for it to run.
    bool checkpoint_due = false;

  private:
    // In WAL frames; 0 for no automatic checkpoints.
    int _wal_autocheckpoint = 0;

    std::thread::id _thread{std::this_thread::get_id()};

    // SQLiteCpp's statements are not thread-safe, so we prepare them thread-locally when needed
    std::unordered_map<
            std::string,
            std::vector<std::unique_ptr<SQLite::Statement>>,
            string_hash,
            std::equal_to<>>
            _st_cache;
};

}  // namespace session::sqlite::detail
