#include "conn.hpp"

#include <sqlite3.h>

#include <session/sqlite.hpp>

namespace session::sqlite::detail {

StatementWrapper conn::prepared_st(std::string_view query) {
    // If you hit this assert hang your head in shame for not following the ridiculous number of
    // times you have been told not to use a Connection across threads:
    assert(std::this_thread::get_id() == _thread);

    std::unique_ptr<SQLite::Statement> st;
    if (auto it = _st_cache.find(query); it != _st_cache.end() && !it->second.empty()) {
        st = std::move(it->second.back());
        it->second.pop_back();
    } else {
        st = std::make_unique<SQLite::Statement>(
                sql, std::string{query}, SQLite::PREPARE_PERSISTENT);
    }

    return {*this, std::move(st)};
}

void conn::statement_finished(std::unique_ptr<SQLite::Statement> st) {
    if (!st)
        return;
    _st_cache[st->getQuery()].push_back(std::move(st));
}

void conn::install_wal_hook() {
    _wal_autocheckpoint = sql.execAndGet("PRAGMA wal_autocheckpoint").getInt();
    sqlite3_wal_hook(
            sql.getHandle(),
            [](void* self, sqlite3* db, const char* name, int frames) {
                auto& c = *static_cast<conn*>(self);
                if (c._wal_autocheckpoint > 0 && frames >= c._wal_autocheckpoint) {
                    if (c.in_immediate)
                        c.checkpoint_due = true;
                    else
                        sqlite3_wal_checkpoint_v2(
                                db, name, SQLITE_CHECKPOINT_PASSIVE, nullptr, nullptr);
                }
                return SQLITE_OK;
            },
            this);
}

}  // namespace session::sqlite::detail
