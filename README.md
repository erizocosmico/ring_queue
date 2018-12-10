# ring_queue [![Build Status](https://travis-ci.org/erizocosmico/ring_queue.svg?branch=master)](https://travis-ci.org/erizocosmico/ring_queue) [![Documentation](https://docs.rs/ring_queue/badge.svg)](https://docs.rs/ring_queue) [![crates.io](https://img.shields.io/crates/v/ring_queue.svg)](https://crates.io/crates/ring_queue) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A double-ended queue implemented using a `Vec` that reuses space after
elements are removed.

The API is heavily based on `collections.deque` from Python.

You can create a ring using any of the available constructors or the `ring!` macro.

```rust
#[macro_use] extern crate ring_queue;

use ring_queue::Ring;

// `new` for an empty ring.
let r: Ring<i32> = Ring::new();

// `with_capacity` for allocating the internal vector with the given
// capacity.
let r2: Ring<i32> = Ring::with_capacity(5);

// `ring!` macro for easy initializing the ring.
let r3: Ring<i32> = ring![1, 2, 3];

// `from_iter` to construct the ring from an iterator.
use std::iter::FromIterator;
let r4: Ring<i32> = Ring::from_iter(vec![1, 2, 3]);
```

Instead of `front` and `back` as a nomenclature, this library uses `left`
to refer to the front an nothing to refer to the back, as the Python
`collections.deque` library does.

Items can be pushed to the left and right as well as popped.

```rust
#[macro_use] extern crate ring_queue;

use ring_queue::Ring;

let mut r = ring![1, 2, 3];
r.push(4);
r.push_left(0);
assert_eq!(r.pop(), Some(4));
assert_eq!(r.pop_left(), Some(0));
```

The ring can be rotated either to the left or to the right. Any positive
number will rotate `n` steps to the right and any negative number will
rotate `n` steps to the left.

```rust
#[macro_use] extern crate ring_queue;

use ring_queue::Ring;

let mut r = ring![1, 2, 3, 4, 5];

r.rotate(1);
assert_eq!(r.collect(), vec![5, 1, 2, 3, 4]);

r.rotate(-2);
assert_eq!(r.collect(), vec![2, 3, 4, 5, 1]);
```

Ring implements `collect` to collect the elements in the ring as a vector
if the type of the elements implements the `Copy` trait.
It also implements `into_iter` to generate an iterator. However,
`into_iter` empties the ring.

```rust
#[macro_use] extern crate ring_queue;

use ring_queue::Ring;

let mut r = ring![1, 2, 3, 4];
assert_eq!(r.collect(), vec![1, 2, 3, 4]);
assert_eq!(r.is_empty(), false);

for item in r.into_iter() {
    println!("{}", item);
}

assert_eq!(r.is_empty(), true);
```

## LICENSE

MIT License, see [LICENSE](/LICENSE)
