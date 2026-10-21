#include <sqlite3.h>

#include <atomic>
#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <filesystem>
#include <latch>
#include <random>
#include <session/sqlite.hpp>
#include <string>
#include <thread>

using namespace session::sqlite;
using namespace std::literals;

namespace {

// A database file in the temp directory, removed (with its WAL and shared-memory files) afterwards.
struct TempPath {
    std::filesystem::path path;
    TempPath() :
            path{std::filesystem::temp_directory_path() /
                 ("session-sqlite-suspend-" + std::to_string(std::random_device{}()) + ".db")} {}
    ~TempPath() {
        std::error_code ec;
        for (const auto* suffix : {"", "-wal", "-shm"})
            std::filesystem::remove(path.string() + suffix, ec);
    }
};

// What another process would see: a connection of its own, outside the pool, that never waits.
// Getting an exclusive lock and truncating the WAL both fail at once while anything -- a reader
// included -- still has the database open with a lock.
struct Outsider {
    sqlite3* db = nullptr;
    explicit Outsider(const std::filesystem::path& path) {
        REQUIRE(sqlite3_open_v2(path.c_str(), &db, SQLITE_OPEN_READWRITE, nullptr) == SQLITE_OK);
        sqlite3_busy_timeout(db, 0);
    }
    ~Outsider() { sqlite3_close(db); }
    bool exclusive() {
        bool ok = sqlite3_exec(db, "BEGIN EXCLUSIVE", nullptr, nullptr, nullptr) == SQLITE_OK;
        if (ok)
            sqlite3_exec(db, "ROLLBACK", nullptr, nullptr, nullptr);
        return ok;
    }
    // A connection holds the database open -- the WAL included -- only once it has read from it.
    void touch() {
        REQUIRE(sqlite3_exec(db, "SELECT count(*) FROM sqlite_master", nullptr, nullptr, nullptr) ==
                SQLITE_OK);
    }
    bool truncate() {
        int log = -1, done = -1;
        return sqlite3_wal_checkpoint_v2(db, nullptr, SQLITE_CHECKPOINT_TRUNCATE, &log, &done) ==
               SQLITE_OK;
    }
};

void make_table(Database& db) {
    auto c = db.conn();
    c.sql.exec("CREATE TABLE foo (x INTEGER)");
    c.sql.exec("INSERT INTO foo VALUES (42)");
}

}  // namespace

TEST_CASE("suspend closes every connection and refuses new ones", "[suspend]") {
    TempPath tmp;
    Database db{tmp.path};
    make_table(db);
    // A second idle connection too, so there is more than one to close.
    {
        auto a = db.conn();
        auto b = db.unique_conn();
    }

    auto result = db.suspend(0ms);
    CHECK(result.closed);
    CHECK_FALSE(result.checkpointed);
    CHECK(db.suspended());
    CHECK_THROWS_AS(db.conn(), database_suspended);
    CHECK_THROWS_AS(db.unique_conn(), database_suspended);

    Outsider other{tmp.path};
    CHECK(other.exclusive());
    CHECK(other.truncate());

    db.resume();
    CHECK_FALSE(db.suspended());
    CHECK(db.conn().prepared_get<int>("SELECT x FROM foo") == 42);
}

TEST_CASE("suspend waits for a held connection, which can still finish", "[suspend]") {
    TempPath tmp;
    Database db{tmp.path};
    make_table(db);

    std::latch holding{1}, suspending{1};
    std::atomic<bool> finished{false};
    std::thread holder{[&] {
        auto c = db.conn();
        SQLite::Transaction tx{c.sql};
        c.sql.exec("INSERT INTO foo VALUES (1)");
        holding.count_down();
        suspending.wait();
        std::this_thread::sleep_for(100ms);
        // The thread's own connection is still handed back to it while suspended.
        db.conn().sql.exec("INSERT INTO foo VALUES (2)");
        tx.commit();
        finished = true;
    }};

    holding.wait();
    std::thread other{[&] {
        // Let the suspend begin before trying from a thread holding nothing.
        std::this_thread::sleep_for(20ms);
        CHECK_THROWS_AS(db.conn(), database_suspended);
    }};
    suspending.count_down();
    auto result = db.suspend(5s);
    holder.join();
    other.join();

    CHECK(result.closed);
    CHECK(finished);
    CHECK(Outsider{tmp.path}.exclusive());

    db.resume();
    CHECK(db.conn().prepared_get<int>("SELECT count(*) FROM foo") == 3);
}

TEST_CASE("a held connection not returned in time is closed when it is", "[suspend]") {
    TempPath tmp;
    Database db{tmp.path};
    make_table(db);

    {
        auto c = db.conn();
        c.sql.exec("BEGIN");
        c.sql.exec("SELECT * FROM foo");
        CHECK_FALSE(db.suspend(50ms).closed);
        c.sql.exec("COMMIT");
    }
    // Returned while suspended, so closed rather than pooled.
    CHECK(Outsider{tmp.path}.exclusive());
    db.resume();
}

TEST_CASE("suspend with interrupt stops a running statement", "[suspend]") {
    TempPath tmp;
    Database db{tmp.path};
    make_table(db);

    std::latch running{1};
    std::atomic<bool> interrupted{false};
    std::thread holder{[&] {
        auto c = db.conn();
        // Never ends on its own.
        SQLite::Statement st{
                c.sql,
                "WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n) "
                "SELECT count(*) FROM n"};
        running.count_down();
        try {
            st.executeStep();
        } catch (const SQLite::Exception& e) {
            interrupted = (e.getErrorCode() & 0xff) == SQLITE_INTERRUPT;
        }
    }};
    running.wait();
    // Give the statement time to start stepping.
    std::this_thread::sleep_for(50ms);
    auto result = db.suspend(5s, true);
    holder.join();

    CHECK(interrupted);
    CHECK(result.closed);
    db.resume();
}

TEST_CASE("suspend can checkpoint and truncate the WAL", "[suspend]") {
    // Closing the last connection to a database checkpoints and removes the WAL by itself.  What
    // the checkpoint is for is another process -- the app's extension -- still having the database
    // open, so that this process's closing is not the last.
    TempPath tmp;
    Database db{tmp.path};
    make_table(db);
    Outsider other{tmp.path};
    other.touch();
    auto wal = std::filesystem::path{tmp.path.string() + "-wal"};
    auto fill = [&] {
        auto c = db.conn();
        for (int i = 0; i < 200; i++)
            c.prepared_exec("INSERT INTO foo VALUES (?)", i);
    };

    fill();
    REQUIRE(std::filesystem::file_size(wal) > 0);
    auto plain = db.suspend(5s);
    CHECK(plain.closed);
    CHECK_FALSE(plain.checkpointed);
    CHECK(std::filesystem::file_size(wal) > 0);
    db.resume();

    fill();
    auto result = db.suspend(5s, false, true);
    CHECK(result.closed);
    CHECK(result.checkpointed);
    CHECK(std::filesystem::file_size(wal) == 0);
    CHECK(other.exclusive());

    db.resume();
    CHECK(db.conn().prepared_get<int>("SELECT count(*) FROM foo") == 401);
}

TEST_CASE("resume ends a suspend still waiting", "[suspend]") {
    TempPath tmp;
    Database db{tmp.path};
    make_table(db);

    auto held = db.conn();
    Database::suspend_result result;
    std::thread suspender{[&] { result = db.suspend(30s); }};
    std::this_thread::sleep_for(50ms);
    auto start = std::chrono::steady_clock::now();
    db.resume();
    suspender.join();

    CHECK(std::chrono::steady_clock::now() - start < 5s);
    CHECK_FALSE(result.closed);
    CHECK_FALSE(db.suspended());
}

TEST_CASE("an in-memory database survives suspend", "[suspend][memory]") {
    Database db{":memory:"};
    make_table(db);

    CHECK(db.suspend(0ms).closed);
    CHECK_THROWS_AS(db.conn(), database_suspended);
    db.resume();
    CHECK(db.conn().prepared_get<int>("SELECT x FROM foo") == 42);
}

TEST_CASE("a refused connection is reported before it throws", "[suspend]") {
    TempPath tmp;
    Database db{tmp.path};
    int refused = 0;
    db.on_refused([&] { refused++; });

    CHECK_NOTHROW(db.conn());
    CHECK(refused == 0);
    db.suspend(0ms);
    CHECK_THROWS_AS(db.conn(), database_suspended);
    CHECK_THROWS_AS(db.unique_conn(), database_suspended);
    CHECK(refused == 2);
    db.resume();
    CHECK_NOTHROW(db.conn());
    CHECK(refused == 2);
}
