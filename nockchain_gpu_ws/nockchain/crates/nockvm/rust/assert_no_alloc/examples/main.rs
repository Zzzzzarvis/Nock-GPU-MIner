use assert_no_alloc::*;

#[cfg(debug_assertions)]
#[global_allocator]
static A: AllocDisabler = AllocDisabler;

fn main() {
    println!("Alloc is allowed. Let's allocate some memory...");
    let vec_can_allocate = vec![42; 10];
    println!("This will be executed if the above allocation succeeds: {vec_can_allocate:?}");

    println!();

    let fib5 = assert_no_alloc(|| {
        println!("Alloc is forbidden. Let's calculate something without memory allocations...");

        fn fib(n: u32) -> u32 {
            if n <= 1 {
                1
            } else {
                fib(n - 1) + fib(n - 2)
            }
        }

        fib(5)
    });
    println!("\tSuccess, the 5th fibonacci number is {}", fib5);
    println!();

    assert_no_alloc(|| {
        println!("Alloc is forbidden. Let's allocate some memory...");
        let vec_cannot_allocate = vec![42; 100];
        println!("This will not be executed if the above allocation has aborted. {vec_cannot_allocate:?}");
    });

    println!("This will not be executed if the above allocation has aborted.");
}
