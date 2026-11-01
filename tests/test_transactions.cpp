#include <sqlite3.h>

#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <filesystem>
#include <future>
#include <mutex>
#include <random>
#include <session/sqlite.hpp>
#include <string>
#include <thread>
#include <vector>

using namespace session::sqlite;
using namespace std::literals;

namespace {

struct TempPath {
    std::filesystem::path path;
    TempPath() :
            path{std::filesystem::temp_directory_path() /
                 ("session-sqlite-tx-" + std::to_string(std::random_device{}()) + ".db")} {}
    ~TempPath() {
        std::error_code ec;
        for (const auto* suffix : {"", "-wal", "-shm"})
            std::filesystem::remove(path.string() + suffix, ec);
    }
};

// Whether another connection, which does not wait, can take the write lock right now
bool can_write_elsewhere(const std::filesystem::path& path) {
    sqlite3* other = nullptr;
    REQUIRE(sqlite3_open_v2(path.c_str(), &other, SQLITE_OPEN_READWRITE, nullptr) == SQLITE_OK);
    sqlite3_busy_timeout(other, 0);
    bool ok = sqlite3_exec(other, "BEGIN IMMEDIATE", nullptr, nullptr, nullptr) == SQLITE_OK;
    if (ok)
        sqlite3_exec(other, "ROLLBACK", nullptr, nullptr, nullptr);
    sqlite3_close(other);
    return ok;
}

// Holds the write lock from a connection outside any Database, so outside its in-process lock
struct OutsideWriter {
    sqlite3* db = nullptr;
    explicit OutsideWriter(const std::filesystem::path& path) {
        REQUIRE(sqlite3_open_v2(path.c_str(), &db, SQLITE_OPEN_READWRITE, nullptr) == SQLITE_OK);
        REQUIRE(sqlite3_exec(db, "BEGIN IMMEDIATE", nullptr, nullptr, nullptr) == SQLITE_OK);
    }
    ~OutsideWriter() {
        sqlite3_exec(db, "ROLLBACK", nullptr, nullptr, nullptr);
        sqlite3_close(db);
    }
};

bool is_busy(const SQLite::Exception& e) {
    return (e.getErrorCode() & 0xff) == SQLITE_BUSY;
}

// Whether `f` throws a SQLite::Exception with SQLITE_MISUSE
template <typename F>
bool throws_misuse(F&& f) {
    try {
        f();
    } catch (const SQLite::Exception& e) {
        return (e.getErrorCode() & 0xff) == SQLITE_MISUSE;
    }
    return false;
}

// Whether begin_immediate() from a fresh thread gets the write lock without waiting
bool can_begin_immediate_elsewhere(Database& db) {
    return std::async(std::launch::async, [&db] {
               auto c = db.unique_conn();
               try {
                   auto tx = c.begin_immediate();
                   return true;
               } catch (const SQLite::Exception&) {
                   return false;
               }
           }).get();
}

}  // namespace

TEST_CASE("begin_immediate takes the write lock at BEGIN, begin does not", "[transactions]") {
    TempPath tmp;
    Database db{tmp.path};
    db.conn().sql.exec("CREATE TABLE foo (x INTEGER)");

    {
        auto c = db.conn();
        auto tx = c.begin();
        CHECK(can_write_elsewhere(tmp.path));
    }
    {
        auto c = db.conn();
        auto tx = c.begin_immediate();
        // Before any statement has run in it
        CHECK_FALSE(can_write_elsewhere(tmp.path));
        c.sql.exec("INSERT INTO foo VALUES (1)");
        tx.commit();
    }
    CHECK(can_write_elsewhere(tmp.path));
    CHECK(db.conn().prepared_get<int>("SELECT count(*) FROM foo") == 1);
}

TEST_CASE(
        "a deferred transaction cannot write after another connection commits; an immediate one "
        "waits its turn",
        "[transactions]") {
    // Why a transaction that may write wants begin_immediate: a deferred one reads from a snapshot,
    // and once another connection has committed it can never write.
    TempPath tmp;
    Database db{tmp.path, busy_timeout{std::chrono::milliseconds{2000}}};
    db.conn().sql.exec("CREATE TABLE foo (x INTEGER)");

    auto mine = db.unique_conn();
    auto other = db.unique_conn();

    auto tx = mine.begin();
    CHECK(mine.prepared_get<int>("SELECT count(*) FROM foo") == 0);
    other.sql.exec("INSERT INTO foo VALUES (1)");
    try {
        mine.sql.exec("INSERT INTO foo VALUES (2)");
        FAIL("a stale snapshot was allowed to write");
    } catch (const SQLite::Exception& e) {
        CHECK((e.getErrorCode() & 0xff) == SQLITE_BUSY);
    }
    tx.rollback();

    auto itx = mine.begin_immediate();
    CHECK(mine.prepared_get<int>("SELECT count(*) FROM foo") == 1);
    mine.sql.exec("INSERT INTO foo VALUES (2)");
    itx.commit();
    CHECK(other.prepared_get<int>("SELECT count(*) FROM foo") == 2);
}

TEST_CASE(
        "a waiting begin_immediate gets the write lock as soon as the transaction ahead ends",
        "[transactions]") {
    TempPath tmp;
    Database db{tmp.path};
    db.conn().sql.exec("CREATE TABLE foo (x INTEGER)");

    std::chrono::steady_clock::time_point released;
    std::promise<void> holding;
    std::thread holder{[&] {
        auto c = db.conn();
        auto tx = c.begin_immediate();
        holding.set_value();
        // Waiting in SQLite's busy handler instead, the next attempt would come at 528ms.
        std::this_thread::sleep_for(450ms);
        c.sql.exec("INSERT INTO foo VALUES (1)");
        tx.commit();
        released = std::chrono::steady_clock::now();
    }};
    holding.get_future().wait();

    auto c = db.conn();
    auto tx = c.begin_immediate();
    auto acquired = std::chrono::steady_clock::now();
    holder.join();
    CHECK(c.prepared_get<int>("SELECT count(*) FROM foo") == 1);
    CHECK(acquired - released < 25ms);
    tx.commit();
}

TEST_CASE("begin_immediate serves waiting threads in the order they arrived", "[transactions]") {
    TempPath tmp;
    Database db{tmp.path};
    db.conn().sql.exec("CREATE TABLE foo (x INTEGER)");

    std::mutex order_mutex;
    std::vector<int> order;
    auto record = [&](int who) {
        std::lock_guard lock{order_mutex};
        order.push_back(who);
    };

    std::promise<void> holding, release;
    std::thread holder{[&] {
        auto c = db.conn();
        {
            auto tx = c.begin_immediate();
            holding.set_value();
            release.get_future().wait();
            tx.commit();
        }
        // Straight back for another, which puts it behind everyone already waiting
        auto tx = c.begin_immediate();
        record(0);
        tx.commit();
    }};
    holding.get_future().wait();

    std::vector<std::thread> waiters;
    for (int i = 1; i <= 3; i++) {
        waiters.emplace_back([&, i] {
            auto c = db.conn();
            auto tx = c.begin_immediate();
            record(i);
            tx.commit();
        });
        // So that each is waiting before the next one starts
        std::this_thread::sleep_for(50ms);
    }
    release.set_value();
    holder.join();
    for (auto& t : waiters)
        t.join();
    CHECK(order == std::vector{1, 2, 3, 0});
}

TEST_CASE(
        "the write lock observer is told how long slow transactions waited and held",
        "[transactions]") {
    TempPath tmp;
    Database db{tmp.path};
    db.conn().sql.exec("CREATE TABLE foo (x INTEGER)");

    std::mutex reports_mutex;
    std::vector<std::pair<std::chrono::nanoseconds, std::chrono::nanoseconds>> reports;
    db.set_write_lock_observer(50ms, [&](auto waited, auto held) {
        std::lock_guard lock{reports_mutex};
        reports.emplace_back(waited, held);
    });

    // Neither waits nor holds for long enough to be reported
    db.conn().begin_immediate().commit();

    std::promise<void> holding;
    std::thread holder{[&] {
        auto c = db.conn();
        auto tx = c.begin_immediate();
        holding.set_value();
        std::this_thread::sleep_for(100ms);
        tx.commit();
    }};
    holding.get_future().wait();
    {
        auto c = db.conn();
        auto tx = c.begin_immediate();
        // Reported even when rolled back by destruction
    }
    holder.join();

    REQUIRE(reports.size() == 2);
    // The two threads report after releasing the lock, so in either order
    if (reports[0].second < reports[1].second)
        std::swap(reports[0], reports[1]);
    auto& [holder_waited, holder_held] = reports[0];
    auto& [waiter_waited, waiter_held] = reports[1];
    CHECK(holder_waited < 50ms);
    CHECK(holder_held >= 100ms);
    CHECK(waiter_waited >= 50ms);
    CHECK(waiter_held < 50ms);

    // Reported for the two together reaching the threshold, though neither does alone; the
    // holder, with only its 30ms hold, is not.
    reports.clear();
    std::promise<void> holding2;
    std::thread holder2{[&] {
        auto c = db.conn();
        auto tx = c.begin_immediate();
        holding2.set_value();
        std::this_thread::sleep_for(30ms);
        tx.commit();
    }};
    holding2.get_future().wait();
    {
        auto c = db.conn();
        auto tx = c.begin_immediate();
        std::this_thread::sleep_for(30ms);
        tx.commit();
    }
    holder2.join();
    REQUIRE(reports.size() == 1);
    CHECK(reports[0].first < 50ms);
    CHECK(reports[0].second < 50ms);
    CHECK(reports[0].first + reports[0].second >= 50ms);
}

TEST_CASE("begin_immediate throws SQLITE_BUSY when the busy timeout runs out", "[transactions]") {
    TempPath tmp;
    Database db{tmp.path, busy_timeout{200ms}};

    auto expect_timeout = [&] {
        auto c = db.conn();
        auto start = std::chrono::steady_clock::now();
        try {
            auto tx = c.begin_immediate();
            FAIL("begin_immediate got the write lock while it was held");
        } catch (const SQLite::Exception& e) {
            CHECK(is_busy(e));
        }
        auto waited = std::chrono::steady_clock::now() - start;
        CHECK(waited >= 200ms);
        CHECK(waited < 1s);
    };

    SECTION("held by another thread's begin_immediate") {
        std::promise<void> holding, done;
        std::thread holder{[&] {
            auto c = db.conn();
            auto tx = c.begin_immediate();
            holding.set_value();
            done.get_future().wait();
        }};
        holding.get_future().wait();
        expect_timeout();
        done.set_value();
        holder.join();
    }
    SECTION("held by another process") {
        OutsideWriter outside{tmp.path};
        expect_timeout();
    }
}

TEST_CASE(
        "the busy timeout covers waiting for another thread and for SQLite together",
        "[transactions]") {
    TempPath tmp;
    Database db{tmp.path, busy_timeout{400ms}};
    db.conn().sql.exec("CREATE TABLE foo (x INTEGER)");
    OutsideWriter outside{tmp.path};

    // Holds the in-process lock while its BEGIN waits in SQLite's busy handler for the outside
    // writer, until that wait times out.
    auto first = std::async(std::launch::async, [&db] {
        auto c = db.conn();
        try {
            auto tx = c.begin_immediate();
            return false;
        } catch (const SQLite::Exception& e) {
            return is_busy(e);
        }
    });
    std::this_thread::sleep_for(50ms);

    auto c = db.conn();
    auto start = std::chrono::steady_clock::now();
    try {
        auto tx = c.begin_immediate();
        FAIL("begin_immediate got the write lock while it was held");
    } catch (const SQLite::Exception& e) {
        CHECK(is_busy(e));
    }
    // ~400ms in all; given a fresh 400ms for its own BEGIN after waiting for `first` it would be
    // ~750ms.
    CHECK(std::chrono::steady_clock::now() - start < 550ms);
    CHECK(first.get());

    // The connection's own busy timeout is back to the full 400ms for anything after.
    start = std::chrono::steady_clock::now();
    CHECK_THROWS_AS(c.sql.exec("INSERT INTO foo VALUES (1)"), SQLite::Exception);
    CHECK(std::chrono::steady_clock::now() - start >= 350ms);
}

TEST_CASE("an ImmediateTransaction releases the write lock when it ends", "[transactions]") {
    TempPath tmp;
    // No waiting at all, so anything still held fails at once
    Database db{tmp.path, busy_timeout{0ms}};
    db.conn().sql.exec("CREATE TABLE foo (x INTEGER)");
    auto c = db.conn();

    SECTION("by commit") {
        auto tx = c.begin_immediate();
        c.sql.exec("INSERT INTO foo VALUES (1)");
        CHECK_FALSE(can_begin_immediate_elsewhere(db));
        tx.commit();
        CHECK(can_begin_immediate_elsewhere(db));
        CHECK(can_write_elsewhere(tmp.path));
        CHECK(c.prepared_get<int>("SELECT count(*) FROM foo") == 1);
        CHECK_THROWS_AS(tx.commit(), SQLite::Exception);
    }
    SECTION("by rollback") {
        auto tx = c.begin_immediate();
        c.sql.exec("INSERT INTO foo VALUES (1)");
        tx.rollback();
        CHECK(can_begin_immediate_elsewhere(db));
        CHECK(can_write_elsewhere(tmp.path));
        CHECK(c.prepared_get<int>("SELECT count(*) FROM foo") == 0);
        CHECK_THROWS_AS(tx.commit(), SQLite::Exception);
    }
    SECTION("by being destroyed, rolling back") {
        {
            auto tx = c.begin_immediate();
            c.sql.exec("INSERT INTO foo VALUES (1)");
        }
        CHECK(can_begin_immediate_elsewhere(db));
        CHECK(can_write_elsewhere(tmp.path));
        CHECK(c.prepared_get<int>("SELECT count(*) FROM foo") == 0);
    }
    SECTION("by failing to BEGIN") {
        {
            OutsideWriter outside{tmp.path};
            try {
                auto tx = c.begin_immediate();
                FAIL("begin_immediate got the write lock while it was held");
            } catch (const SQLite::Exception& e) {
                CHECK(is_busy(e));
            }
        }
        CHECK(can_begin_immediate_elsewhere(db));
        auto tx = c.begin_immediate();
        tx.commit();
    }
}

TEST_CASE(
        "begin_immediate refuses a thread that already has one open on the same Database",
        "[transactions]") {
    TempPath tmp;
    Database db{tmp.path};
    db.conn().sql.exec("CREATE TABLE foo (x INTEGER)");

    auto c = db.conn();
    auto tx = c.begin_immediate();
    auto other = db.unique_conn();
    CHECK(throws_misuse([&] { (void)c.begin_immediate(); }));
    CHECK(throws_misuse([&] { (void)other.begin_immediate(); }));

    // The one that is open is unaffected
    c.sql.exec("INSERT INTO foo VALUES (1)");
    tx.commit();

    auto tx2 = other.begin_immediate();
    other.sql.exec("INSERT INTO foo VALUES (2)");
    tx2.commit();
    CHECK(c.prepared_get<int>("SELECT count(*) FROM foo") == 2);
}

TEST_CASE("begin_immediate waits for the timeout it is given", "[transactions]") {
    TempPath tmp;

    // Throws SQLITE_BUSY from `f`, returning how long it took
    auto busy_after = [](auto&& f) {
        auto start = std::chrono::steady_clock::now();
        try {
            f();
            FAIL("got the write lock while it was held");
        } catch (const SQLite::Exception& e) {
            CHECK(is_busy(e));
        }
        return std::chrono::steady_clock::now() - start;
    };

    // Holds the write lock through `db` from another thread until destroyed
    struct Holder {
        std::promise<void> holding, done;
        std::thread thread;
        Holder(Database& db, std::chrono::milliseconds hold = 1h) {
            thread = std::thread{[this, &db, hold] {
                auto c = db.conn();
                auto tx = c.begin_immediate();
                holding.set_value();
                done.get_future().wait_for(hold);
            }};
            holding.get_future().wait();
        }
        ~Holder() {
            done.set_value();
            thread.join();
        }
    };

    SECTION("shorter than the busy timeout") {
        Database db{tmp.path, busy_timeout{5s}};
        Holder holder{db};
        auto c = db.conn();
        auto waited = busy_after([&] { (void)c.begin_immediate(100ms); });
        CHECK(waited >= 100ms);
        CHECK(waited < 1s);
    }
    SECTION("longer than the busy timeout") {
        Database db{tmp.path, busy_timeout{100ms}};
        Holder holder{db, 300ms};
        auto c = db.conn();
        auto tx = c.begin_immediate(5s);
        tx.commit();
    }
    SECTION("0, taking the lock only if it is free now") {
        Database db{tmp.path, busy_timeout{5s}};
        auto c = db.conn();
        c.begin_immediate(0ms).commit();
        Holder holder{db};
        CHECK(busy_after([&] { (void)c.begin_immediate(0ms); }) < 50ms);
    }
    SECTION("while another process holds the lock") {
        Database db{tmp.path, busy_timeout{1s}};
        db.conn().sql.exec("CREATE TABLE foo (x INTEGER)");
        auto c = db.conn();
        OutsideWriter outside{tmp.path};
        auto waited = busy_after([&] { (void)c.begin_immediate(200ms); });
        CHECK(waited >= 200ms);
        CHECK(waited < 800ms);
        CHECK(busy_after([&] { (void)c.begin_immediate(0ms); }) < 50ms);
        // The connection's own busy timeout is back for anything after
        CHECK(busy_after([&] { c.sql.exec("INSERT INTO foo VALUES (1)"); }) >= 900ms);
    }
    SECTION("in prepared_exec_immediate") {
        Database db{tmp.path, busy_timeout{5s}};
        db.conn().sql.exec("CREATE TABLE foo (x INTEGER)");
        Holder holder{db};
        auto c = db.conn();
        auto waited = busy_after(
                [&] { c.prepared_exec_immediate(100ms, "INSERT INTO foo VALUES (?)", 1); });
        CHECK(waited >= 100ms);
        CHECK(waited < 1s);
    }
}

TEST_CASE("prepared_exec_immediate writes in a begin_immediate transaction", "[transactions]") {
    TempPath tmp;
    Database db{tmp.path, busy_timeout{0ms}};
    db.conn().sql.exec("CREATE TABLE foo (x INTEGER)");
    auto c = db.conn();

    CHECK(c.prepared_exec_immediate("INSERT INTO foo VALUES (?), (?)", 1, 2) == 2);
    CHECK(c.prepared_get<int>("SELECT count(*) FROM foo") == 2);
    CHECK(can_begin_immediate_elsewhere(db));

    {
        auto tx = c.begin_immediate();
        CHECK(throws_misuse([&] { c.prepared_exec_immediate("INSERT INTO foo VALUES (3)"); }));
    }

    std::promise<void> holding, done;
    std::thread holder{[&] {
        auto c = db.conn();
        auto tx = c.begin_immediate();
        holding.set_value();
        done.get_future().wait();
    }};
    holding.get_future().wait();
    try {
        c.prepared_exec_immediate("INSERT INTO foo VALUES (4)");
        FAIL("prepared_exec_immediate wrote while another thread held the write lock");
    } catch (const SQLite::Exception& e) {
        CHECK(is_busy(e));
    }
    done.set_value();
    holder.join();
    CHECK(c.prepared_get<int>("SELECT count(*) FROM foo") == 2);
}

TEST_CASE("the WAL is still checkpointed at the connection's threshold", "[transactions]") {
    TempPath tmp;
    auto wal_size = [&] { return std::filesystem::file_size(tmp.path.string() + "-wal"); };
    const auto blob = std::string(3000, 'x');
    constexpr int threshold = 50, writes = 300;

    SECTION("after begin_immediate() transactions") {
        Database db{tmp.path, post_open{[](Connection& c) {
                        c.sql.exec("PRAGMA wal_autocheckpoint = " + std::to_string(threshold));
                    }}};
        auto c = db.conn();
        c.sql.exec("CREATE TABLE foo (x BLOB)");
        for (int i = 0; i < writes; i++) {
            auto tx = c.begin_immediate();
            c.prepared_exec("INSERT INTO foo VALUES (?)", blob);
            tx.commit();
        }
        // A checkpointed WAL is reused from the start, so it never gets much past the threshold
        CHECK(wal_size() < 2 * threshold * 4096);
    }
    SECTION("after writes made without begin_immediate()") {
        Database db{tmp.path, post_open{[](Connection& c) {
                        c.sql.exec("PRAGMA wal_autocheckpoint = " + std::to_string(threshold));
                    }}};
        auto c = db.conn();
        c.sql.exec("CREATE TABLE foo (x BLOB)");
        for (int i = 0; i < writes; i++)
            c.prepared_exec("INSERT INTO foo VALUES (?)", blob);
        CHECK(wal_size() < 2 * threshold * 4096);
    }
    SECTION("not at all with a threshold of 0") {
        Database db{tmp.path, post_open{[](Connection& c) {
                        c.sql.exec("PRAGMA wal_autocheckpoint = 0");
                    }}};
        auto c = db.conn();
        c.sql.exec("CREATE TABLE foo (x BLOB)");
        for (int i = 0; i < writes; i++) {
            auto tx = c.begin_immediate();
            c.prepared_exec("INSERT INTO foo VALUES (?)", blob);
            tx.commit();
        }
        CHECK(wal_size() > writes * 3000);
    }
}
