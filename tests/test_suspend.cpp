#include <sqlite3.h>

#include <atomic>
#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <filesystem>
#include <random>
#include <session/sqlite.hpp>
#include <string>
#include <thread>

using namespace session::sqlite;
using namespace std::literals;

namespace {

// A one-shot signal between threads, waited on by spinning: a latch is not available on the macOS
// deployment target.
struct Gate {
    std::atomic<bool> open{false};
    void count_down() { open = true; }
    void wait() const {
        while (!open)
            std::this_thread::yield();
    }
};

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

    REQUIRE(db.open_connections() == 2);
    auto result = db.suspend(0ms);
    CHECK(result.closed);
    CHECK_FALSE(result.checkpointed);
    CHECK(db.suspended());
    // What the outsider below cannot tell apart: an idle connection still open holds a lock on the
    // shared-memory file without stopping its exclusive lock or its checkpoint.
    CHECK(db.open_connections() == 0);
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

    Gate holding, suspending;
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

    Gate running;
    std::atomic<bool> interrupted{false};
    std::thread holder{[&] {
        auto c = db.conn();
        // Never ends on its own.
        SQLite::Statement st{
                c.sql,
                "WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n) "
                "SELECT count(*) FROM n"};
        // Called from inside the running statement, so the interrupt cannot land before it starts
        // -- where it would do nothing and the statement would run for ever.
        std::atomic<bool> started{false};
        struct Progress {
            Gate& running;
            std::atomic<bool>& started;
        } progress{running, started};
        sqlite3_progress_handler(
                c.sql.getHandle(),
                1000,
                [](void* p) {
                    auto& [running, started] = *static_cast<Progress*>(p);
                    if (!started.exchange(true))
                        running.count_down();
                    return 0;
                },
                &progress);
        try {
            st.executeStep();
        } catch (const SQLite::Exception& e) {
            interrupted = (e.getErrorCode() & 0xff) == SQLITE_INTERRUPT;
        }
        sqlite3_progress_handler(c.sql.getHandle(), 0, nullptr, nullptr);
    }};
    running.wait();
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

    // The checkpoint's own connection is not left behind in the pool.
    CHECK(db.open_connections() == 0);

    db.resume();
    CHECK(db.conn().prepared_get<int>("SELECT count(*) FROM foo") == 401);
}

TEST_CASE("a checkpoint the other process blocks gives up within the wait", "[suspend]") {
    // Opening a connection reads the database, and the pool's busy timeout is far longer than the
    // wait: the checkpoint's connection has to be opened within the wait too.
    TempPath tmp;
    Database db{tmp.path, busy_timeout{10s}};
    make_table(db);
    Outsider other{tmp.path};
    REQUIRE(sqlite3_exec(other.db, "BEGIN EXCLUSIVE", nullptr, nullptr, nullptr) == SQLITE_OK);
    REQUIRE(sqlite3_exec(other.db, "INSERT INTO foo VALUES (1)", nullptr, nullptr, nullptr) ==
            SQLITE_OK);

    auto start = std::chrono::steady_clock::now();
    Database::suspend_result result;
    CHECK_NOTHROW(result = db.suspend(200ms, false, true));
    CHECK(std::chrono::steady_clock::now() - start < 3s);
    CHECK(result.closed);
    CHECK_FALSE(result.checkpointed);
    CHECK(db.open_connections() == 0);

    sqlite3_exec(other.db, "ROLLBACK", nullptr, nullptr, nullptr);
    db.resume();
    CHECK(db.conn().prepared_get<int>("SELECT count(*) FROM foo") == 1);
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
    CHECK(db.open_connections() == 1);
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

TEST_CASE("a held thread waits for the resume instead of being refused", "[suspend][hold]") {
    TempPath tmp;
    Database db{tmp.path};
    make_table(db);
    // Written by the held thread under the pool's lock, so read here only once it is joined --
    // except the count, which says when it has begun waiting.
    std::atomic<int> holds{0};
    std::vector<std::chrono::milliseconds> waited;
    db.on_hold(
            [&](auto w) {
                waited.push_back(w);
                holds++;
            },
            40ms);
    int refused = 0;
    db.on_refused([&] { refused++; });

    Gate registered;
    std::atomic<int> got{0};
    std::atomic<bool> stop{false};
    std::thread held{[&] {
        db.hold_while_suspended(std::this_thread::get_id());
        registered.count_down();
        while (!stop) {
            got = db.conn().prepared_get<int>("SELECT x FROM foo");
            std::this_thread::sleep_for(5ms);
        }
        db.stop_holding(std::this_thread::get_id());
    }};
    registered.wait();

    REQUIRE(db.suspend(1s).closed);
    for (int i = 0; i < 200 && holds == 0; i++)
        std::this_thread::sleep_for(5ms);
    REQUIRE(holds > 0);
    got = 0;
    std::this_thread::sleep_for(150ms);
    CHECK(got == 0);
    CHECK(db.open_connections() == 0);
    // Other threads are still refused.
    CHECK_THROWS_AS(db.conn(), database_suspended);
    CHECK(refused == 1);

    db.resume();
    for (int i = 0; i < 200 && got == 0; i++)
        std::this_thread::sleep_for(5ms);
    stop = true;
    held.join();
    CHECK(got == 42);
    REQUIRE(waited.size() >= 3);
    CHECK(waited[0] == 0ms);
    CHECK(waited[1] >= 40ms);
}

TEST_CASE("a thread no longer held is refused again", "[suspend][hold]") {
    TempPath tmp;
    Database db{tmp.path};
    auto me = std::this_thread::get_id();
    db.hold_while_suspended(me);
    db.stop_holding(me);
    db.suspend(0ms);
    CHECK_THROWS_AS(db.conn(), database_suspended);
    db.resume();
}

TEST_CASE(
        "a checkpoint and a connection after a resume open an encrypted database together",
        "[suspend][hold]") {
    // The checkpoint's connection is opened without the pool's lock, so that a resume can go
    // ahead meanwhile - and a held thread released by that resume opens one of its own at the same
    // time.  Both read the key, whose buffer is unlocked and relocked around each read.
    if (!enabled(Encryption::AEGIS256))
        SKIP("AEGIS256 not supported by this build");
    TempPath tmp;
    std::array<std::byte, 32> key;
    for (size_t i = 0; i < key.size(); ++i)
        key[i] = static_cast<std::byte>(i * 7 + 1);
    Database db{tmp.path, Encryption::AEGIS256, raw_key{key}};
    make_table(db);
    db.on_hold([](auto) {});

    std::atomic<bool> stop{false};
    Gate registered;
    std::thread held{[&] {
        db.hold_while_suspended(std::this_thread::get_id());
        registered.count_down();
        while (!stop)
            CHECK(db.unique_conn().prepared_get<int>("SELECT x FROM foo") == 42);
        db.stop_holding(std::this_thread::get_id());
    }};
    registered.wait();

    for (int i = 0; i < 50; ++i) {
        std::thread suspender{[&] { db.suspend(1s, false, true); }};
        std::this_thread::sleep_for(std::chrono::microseconds{100 * (i % 7)});
        db.resume();
        suspender.join();
        db.resume();
    }
    stop = true;
    held.join();
}

namespace {

bool is_interrupt(const SQLite::Exception& e) {
    return (e.getErrorCode() & 0xff) == SQLITE_INTERRUPT;
}

// Holds the write lock in a begin_immediate() transaction on a thread of its own, inserting `x`,
// until released; then commits.
struct ImmediateHolder {
    Gate holding, release;
    std::thread thread;
    ImmediateHolder(Database& db, int x) :
            thread{[&db, x, this] {
                auto c = db.conn();
                auto tx = c.begin_immediate();
                c.prepared_exec("INSERT INTO foo VALUES (?)", x);
                holding.count_down();
                release.wait();
                tx.commit();
            }} {
        holding.wait();
    }
    void finish() {
        release.count_down();
        if (thread.joinable())
            thread.join();
    }
    ~ImmediateHolder() { finish(); }
};

}  // namespace

TEST_CASE("a begin_immediate waiter carries on across a suspend", "[suspend][transactions]") {
    TempPath tmp;
    Database db{tmp.path};
    make_table(db);

    ImmediateHolder holder{db, 1};
    // Holds its connection from before the suspend, so it keeps it, and waits in the queue for the
    // write lock with a timeout far longer than any wait below.
    Gate queuing;
    std::atomic<bool> acquired{false}, committed{false};
    std::thread waiter{[&] {
        auto c = db.conn();
        queuing.count_down();
        auto tx = c.begin_immediate(30s);
        acquired = true;
        c.prepared_exec("INSERT INTO foo VALUES (2)");
        tx.commit();
        committed = true;
    }};
    queuing.wait();
    std::this_thread::sleep_for(50ms);

    SECTION("resumed before the suspend could close its connection") {
        CHECK_FALSE(db.suspend(100ms).closed);
        CHECK_FALSE(acquired);
        db.resume();
        holder.finish();
        waiter.join();
    }
    SECTION("finishing before the suspend closes its connection") {
        Database::suspend_result result;
        std::thread suspender{[&] { result = db.suspend(10s); }};
        std::this_thread::sleep_for(50ms);
        CHECK_FALSE(acquired);
        // The holder's release hands the lock straight to the queued waiter, which can finish
        // what it started and so let the suspend close its connection.
        holder.finish();
        waiter.join();
        suspender.join();
        CHECK(result.closed);
        CHECK(db.open_connections() == 0);
        db.resume();
    }

    CHECK(committed);
    CHECK(db.conn().prepared_get<int>("SELECT count(*) FROM foo WHERE x IN (1, 2)") == 2);
    // The lock is free for whoever comes next.
    CHECK(db.conn().prepared_exec_immediate(0ms, "INSERT INTO foo VALUES (3)") == 1);
}

TEST_CASE(
        "suspend with interrupt makes begin_immediate waiters give up, not the holder",
        "[suspend][transactions]") {
    TempPath tmp;
    Database db{tmp.path};
    make_table(db);
    std::atomic<int> observed{0};
    db.set_write_lock_observer(0ns, [&](auto, auto) { observed++; });

    // Not running a statement, so the interrupt has nothing of the holder's to stop.
    ImmediateHolder holder{db, 1};
    // Counted here and checked on the main thread: a failed assertion must not throw out of a
    // thread.
    std::atomic<int> interrupted{0}, not_interrupted{0};
    Gate queuing, waiting, suspended, later_queuing;
    auto expect_interrupt = [&](auto&& f) {
        try {
            f();
        } catch (const SQLite::Exception& e) {
            if (is_interrupt(e)) {
                interrupted++;
                return;
            }
        }
        not_interrupted++;
    };
    std::thread waiter{[&] {
        auto c = db.conn();
        queuing.count_down();
        expect_interrupt([&] {
            auto tx = c.begin_immediate(30s);
            c.prepared_exec("INSERT INTO foo VALUES (2)");
            tx.commit();
        });
    }};
    queuing.wait();
    std::thread exec_waiter{[&] {
        auto c = db.conn();
        waiting.count_down();
        expect_interrupt([&] { c.prepared_exec_immediate(30s, "INSERT INTO foo VALUES (3)"); });
    }};
    waiting.wait();
    // Has its connection from before the suspend, but asks for the lock only after the interrupt,
    // and so waits for it as usual.
    std::atomic<bool> later_committed{false};
    std::thread later{[&] {
        auto c = db.conn();
        later_queuing.count_down();
        suspended.wait();
        auto tx = c.begin_immediate(30s);
        c.prepared_exec("INSERT INTO foo VALUES (4)");
        tx.commit();
        later_committed = true;
    }};
    later_queuing.wait();
    std::this_thread::sleep_for(50ms);

    auto start = std::chrono::steady_clock::now();
    CHECK_FALSE(db.suspend(200ms, true).closed);
    waiter.join();
    exec_waiter.join();
    CHECK(std::chrono::steady_clock::now() - start < 5s);
    CHECK(interrupted == 2);
    CHECK(not_interrupted == 0);

    suspended.count_down();
    std::this_thread::sleep_for(50ms);
    CHECK_FALSE(later_committed);
    // The holder still has the lock, and commits what it wrote.
    holder.finish();
    later.join();
    CHECK(later_committed);
    CHECK(db.open_connections() == 0);
    db.resume();

    CHECK(db.conn().prepared_get<std::string>(
                  "SELECT group_concat(x) FROM (SELECT x FROM foo ORDER BY x)") == "1,4,42");
    // Transactions that ended are reported, not calls that gave up before taking the lock.
    CHECK(observed == 2);
}

TEST_CASE(
        "a held thread's prepared_exec_immediate waits for the resume",
        "[suspend][hold][transactions]") {
    TempPath tmp;
    Database db{tmp.path};
    make_table(db);
    db.on_hold([](auto) {});

    REQUIRE(db.suspend(0ms).closed);
    std::atomic<bool> written{false};
    std::thread held{[&] {
        db.hold_while_suspended(std::this_thread::get_id());
        db.conn().prepared_exec_immediate("INSERT INTO foo VALUES (1)");
        written = true;
        db.stop_holding(std::this_thread::get_id());
    }};
    std::this_thread::sleep_for(100ms);
    CHECK_FALSE(written);
    CHECK(db.open_connections() == 0);
    db.resume();
    held.join();
    CHECK(written);
    CHECK(db.conn().prepared_get<int>("SELECT count(*) FROM foo") == 2);
}

TEST_CASE("a suspend waits for begin_immediate's deferred checkpoint", "[suspend][transactions]") {
    // The checkpoint a COMMIT leaves for after the write lock is released runs on the connection
    // its thread still holds, so it ends before that connection is returned -- and so before a
    // suspend that waits for every connection returns.  The suspend is told not to checkpoint, so
    // that the rows reaching the main file can only be the deferred checkpoint's doing; `other`
    // keeps the WAL from being checkpointed and removed as the last pool connection closes.
    TempPath tmp, copy;
    constexpr int threshold = 50;
    Database db{tmp.path, post_open{[](Connection& c) {
                    c.sql.exec("PRAGMA wal_autocheckpoint = " + std::to_string(threshold));
                }}};
    make_table(db);
    Outsider other{tmp.path};
    other.touch();

    Gate holding, release;
    std::thread holder{[&] {
        auto c = db.conn();
        auto tx = c.begin_immediate();
        const auto blob = std::string(3000, 'x');
        for (int i = 0; i < 2 * threshold; i++)
            c.prepared_exec("INSERT INTO foo VALUES (?)", blob);
        holding.count_down();
        release.wait();
        tx.commit();
    }};
    holding.wait();

    Database::suspend_result result;
    std::thread suspender{[&] { result = db.suspend(10s, false, false); }};
    std::this_thread::sleep_for(50ms);
    release.count_down();
    holder.join();
    suspender.join();

    CHECK(result.closed);
    CHECK_FALSE(result.checkpointed);
    CHECK(db.open_connections() == 0);

    // The main file alone, without the WAL beside it.
    std::filesystem::copy_file(tmp.path, copy.path);
    Outsider main_only{copy.path};
    sqlite3_stmt* st = nullptr;
    REQUIRE(sqlite3_prepare_v2(main_only.db, "SELECT count(*) FROM foo", -1, &st, nullptr) ==
            SQLITE_OK);
    REQUIRE(sqlite3_step(st) == SQLITE_ROW);
    CHECK(sqlite3_column_int(st, 0) == 2 * threshold + 1);
    sqlite3_finalize(st);
    db.resume();
}
