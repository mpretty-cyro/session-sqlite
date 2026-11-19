#include <sqlite3.h>

#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <filesystem>
#include <future>
#include <random>
#include <session/sqlite.hpp>
#include <string>
#include <thread>

using namespace session::sqlite;
using namespace std::literals;

namespace {

struct TempPath {
    std::filesystem::path path;
    TempPath() :
            path{std::filesystem::temp_directory_path() /
                 ("session-sqlite-int-" + std::to_string(std::random_device{}()) + ".db")} {}
    ~TempPath() {
        std::error_code ec;
        for (const auto* suffix : {"", "-wal", "-shm"})
            std::filesystem::remove(path.string() + suffix, ec);
    }
};

bool is_interrupt(const SQLite::Exception& e) {
    return (e.getErrorCode() & 0xff) == SQLITE_INTERRUPT;
}

// A statement that runs for far longer than any test waits, unless it is stopped.
constexpr auto LONG_STATEMENT =
        "WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM n LIMIT 4000000000) "
        "SELECT count(*) FROM n";

// Runs, on its own thread, a transaction that writes a row and then a long statement, and answers
// how it ended: "committed", "interrupted", or something else.  `started` is set once the row is
// written.
std::future<std::string> long_write(Database& db, std::promise<void>& started) {
    return std::async(std::launch::async, [&db, &started] {
        try {
            auto c = db.conn();
            auto tx = c.begin_immediate();
            c.prepared_exec("INSERT INTO t VALUES (1)");
            started.set_value();
            c.prepared_get<int64_t>(LONG_STATEMENT);
            tx.commit();
            return std::string{"committed"};
        } catch (const SQLite::Exception& e) {
            return std::string{is_interrupt(e) ? "interrupted" : e.what()};
        }
    });
}

int64_t rows(const std::filesystem::path& path) {
    Database other{path};
    auto c = other.conn();
    return c.prepared_get<int64_t>("SELECT count(*) FROM t");
}

void create_table(Database& db) {
    auto c = db.conn();
    c.prepared_exec("CREATE TABLE t (x INTEGER)");
}

}  // namespace

TEST_CASE("interrupting stops a running statement and rolls its transaction back", "[interrupt]") {
    TempPath tmp;
    Database db{tmp.path};
    create_table(db);

    std::promise<void> started;
    auto ended = long_write(db, started);
    started.get_future().wait();
    std::this_thread::sleep_for(50ms);  // well into the long statement

    auto before = std::chrono::steady_clock::now();
    db.interrupt_handle()->interrupt();
    REQUIRE(ended.wait_for(5s) == std::future_status::ready);
    CHECK(ended.get() == "interrupted");
    CHECK(std::chrono::steady_clock::now() - before < 1s);
    CHECK(rows(tmp.path) == 0);
}

TEST_CASE("an interrupted Database refuses every borrow", "[interrupt]") {
    TempPath tmp;
    Database db{tmp.path};
    auto handle = db.interrupt_handle();
    CHECK_FALSE(handle->interrupted());
    handle->interrupt();
    CHECK(handle->interrupted());

    for (auto borrow : {+[](Database& d) { d.conn(); }, +[](Database& d) { d.unique_conn(); }}) {
        bool refused = false;
        try {
            borrow(db);
        } catch (const SQLite::Exception& e) {
            refused = is_interrupt(e);
        }
        CHECK(refused);
    }
}

TEST_CASE("interrupting releases a begin_immediate waiting for the write lock", "[interrupt]") {
    TempPath tmp;
    Database db{tmp.path, busy_timeout{30s}};
    create_table(db);

    // This thread holds the write lock; another waits for it with a long timeout.
    auto holder = db.conn();
    auto held = holder.begin_immediate();
    auto waiter = std::async(std::launch::async, [&db] {
        try {
            auto c = db.unique_conn();
            auto tx = c.begin_immediate();
            return std::string{"took the lock"};
        } catch (const SQLite::Exception& e) {
            return std::string{is_interrupt(e) ? "interrupted" : e.what()};
        }
    });
    // The waiter borrows its connection before the interrupt, then queues for the lock.
    std::this_thread::sleep_for(100ms);
    db.interrupt_handle()->interrupt();
    REQUIRE(waiter.wait_for(5s) == std::future_status::ready);
    CHECK(waiter.get() == "interrupted");
}

TEST_CASE("an interrupt handle does nothing once its Database is gone", "[interrupt]") {
    TempPath tmp;
    std::shared_ptr<InterruptHandle> handle;
    {
        Database db{tmp.path};
        create_table(db);
        handle = db.interrupt_handle();
    }
    handle->interrupt();
    CHECK(handle->interrupted());
}

TEST_CASE(
        "destroying a Database interrupts a connection still borrowed and waits for it",
        "[interrupt]") {
    TempPath tmp;
    std::promise<void> started;
    std::future<std::string> ended;
    auto destroyed_in = std::chrono::nanoseconds{};
    {
        auto db = std::make_unique<Database>(tmp.path);
        create_table(*db);
        ended = long_write(*db, started);
        started.get_future().wait();
        std::this_thread::sleep_for(50ms);
        auto before = std::chrono::steady_clock::now();
        db.reset();
        destroyed_in = std::chrono::steady_clock::now() - before;
    }
    REQUIRE(ended.wait_for(5s) == std::future_status::ready);
    CHECK(ended.get() == "interrupted");
    CHECK(destroyed_in < Database::DESTROY_DRAIN_TIMEOUT);
    CHECK(rows(tmp.path) == 0);
}
