add_subdirectory(Catch2)

add_executable(testAll
    test_interrupt.cpp
    test_memory.cpp
    test_transactions.cpp)

target_link_libraries(testAll PRIVATE session::SQLite Catch2::Catch2WithMain)
