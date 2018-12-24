//! A double-ended queue implemented using a `Vec` that reuses space after
//! elements are removed.
//!
//! The API is heavily based on `collections.deque` from Python.
//!
//! You can create a ring using any of the available constructors or the `ring!` macro.
//!
//! ```rust
//! # #[macro_use] extern crate ring_queue;
//!
//! use ring_queue::Ring;
//!
//! // `new` for an empty ring.
//! let r: Ring<i32> = Ring::new();
//!
//! // `with_capacity` for allocating the internal vector with the given
//! // capacity.
//! let r2: Ring<i32> = Ring::with_capacity(5);
//!
//! // `ring!` macro for easy initializing the ring.
//! let r3: Ring<i32> = ring![1, 2, 3];
//!
//! // `from_iter` to construct the ring from an iterator.
//! use std::iter::FromIterator;
//! let r4: Ring<i32> = Ring::from_iter(vec![1, 2, 3]);
//! ```
//!
//! Instead of `front` and `back` as a nomenclature, this library uses `left`
//! to refer to the front an nothing to refer to the back, as the Python
//! `collections.deque` library does.
//!
//! Items can be pushed to the left and right as well as popped.
//!
//! ```rust
//! # #[macro_use] extern crate ring_queue;
//!
//! use ring_queue::Ring;
//!
//! let mut r = ring![1, 2, 3];
//! r.push(4);
//! r.push_left(0);
//! assert_eq!(r.pop(), Some(4));
//! assert_eq!(r.pop_left(), Some(0));
//! ```
//!
//! The ring can be rotated either to the left or to the right. Any positive
//! number will rotate `n` steps to the right and any negative number will
//! rotate `n` steps to the left.
//!
//! ```rust
//! # #[macro_use] extern crate ring_queue;
//!
//! use ring_queue::Ring;
//!
//! let mut r = ring![1, 2, 3, 4, 5];
//!
//! r.rotate(1);
//! assert_eq!(r.collect_vec(), vec![5, 1, 2, 3, 4]);
//!
//! r.rotate(-2);
//! assert_eq!(r.collect_vec(), vec![2, 3, 4, 5, 1]);
//! ```
//!
//! Ring implements `collect` to collect the elements in the ring as a vector
//! if the type of the elements implement the `Copy` trait.
//!
//! ```rust
//! # #[macro_use] extern crate ring_queue;
//!
//! use ring_queue::Ring;
//!
//! let mut r = ring![1, 2, 3, 4];
//! assert_eq!(r.collect_vec(), vec![1, 2, 3, 4]);
//! ```
//!
//! It also implements `into_iter` to generate an iterator. However,
//! `into_iter` empties the ring unless the elements implement the `Copy` trait.
//!
//! ```rust
//! # #[macro_use] extern crate ring_queue;
//!
//! use ring_queue::Ring;
//!
//! #[derive(Debug)]
//! struct Foo { a: i32 }
//!
//! let mut r = ring![Foo{a: 1}, Foo{a: 2}];
//! assert_eq!(r.is_empty(), false);
//!
//! for item in r.into_iter() {
//!     println!("{:?}", item);
//! }
//!
//! assert_eq!(r.is_empty(), true);
//!
//! let r2 = ring![1, 2, 3, 4];
//! assert_eq!(r2.is_empty(), false);
//!
//! for item in r2.into_iter() {
//!     println!("{}", item);
//! }
//!
//! assert_eq!(r2.is_empty(), false);
//! ```

#[derive(Copy, Clone)]
struct Item<T> {
    val: Option<T>,
    prev: usize,
    next: usize,
}

impl<T> Item<T> {
    fn reverse(&mut self) {
        let prev = self.prev;
        self.prev = self.next;
        self.next = prev;
    }
}

#[macro_export]
macro_rules! ring {
    () => ( $crate::Ring::new() );
    ($($x:expr),*) => (
        {
            let s: Box<[_]>  = Box::new([$($x),*]);
            let mut x = s.into_vec();
            let mut r = Ring::new();
            for i in x.drain(..) {
                r.push(i);
            }
            r
        }
    );
    ($($x:expr,)*) => (ring![$($x),*])
}

/// Double-ended queue implemented with a vector that reuses space.
pub struct Ring<T> {
    cur: Option<usize>,
    items: Vec<Item<T>>,
    gaps: Vec<usize>,
}

impl<T> Ring<T>
where
    T: Sized,
{
    /// Create a new empty `Ring`.
    pub fn new() -> Ring<T> {
        Ring {
            items: Vec::new(),
            cur: None,
            gaps: Vec::new(),
        }
    }

    /// Create a new empty `Ring` with a predefined capacity.
    pub fn with_capacity(capacity: usize) -> Ring<T> {
        Ring {
            items: Vec::with_capacity(capacity),
            cur: None,
            gaps: Vec::new(),
        }
    }

    /// Removes the value on the left side of the ring, that is, the head. It
    /// will return the popped value, if there was any.
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate ring_queue;
    /// # fn main() {
    /// use ring_queue::Ring;
    ///
    /// let mut r = ring![1, 2];
    /// assert_eq!(r.pop_left(), Some(1));
    /// assert_eq!(r.pop_left(), Some(2));
    /// assert_eq!(r.pop_left(), None);
    /// # }
    /// ```
    pub fn pop_left(&mut self) -> Option<T> {
        let mut val = None;
        self.cur = self.cur.and_then(|c| {
            let mut item = Item {
                val: None,
                next: 0,
                prev: 0,
            };
            std::mem::swap(&mut item, &mut self.items[c]);
            val = item.val;

            // Remove the value from the item and push it to the gap list.
            self.gaps.push(c);

            // If the next item is this item, it means there is only one
            // element in the ring, so return an empty cursor.
            if item.next == c {
                None
            } else {
                self.items[item.prev].next = item.next;
                self.items[item.next].prev = item.prev;
                Some(item.next)
            }
        });
        val
    }

    /// Removes the value on the right side of the ring, that is, the tail. It
    /// will return the popped value, if there was any.
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate ring_queue;
    /// # fn main() {
    /// use ring_queue::Ring;
    ///
    /// let mut r = ring![1, 2];
    /// assert_eq!(r.pop(), Some(2));
    /// assert_eq!(r.pop(), Some(1));
    /// assert_eq!(r.pop(), None);
    /// # }
    /// ```
    pub fn pop(&mut self) -> Option<T> {
        match self.cur {
            Some(c) => {
                let head = self.items[c].prev;
                let mut tail = Item {
                    val: None,
                    next: 0,
                    prev: 0,
                };
                std::mem::swap(&mut tail, &mut self.items[head]);
                let val = tail.val;

                // Remove the value from the node and push it to the gap list.
                tail.val = None;
                self.gaps.push(head);

                // If the next node is this node, it means there is only one
                // element in the ring, so return an empty cursor.
                if tail.next != head {
                    self.items[tail.prev].next = tail.next;
                    self.items[tail.next].prev = tail.prev;
                }

                val
            }
            None => None,
        }
    }

    #[inline]
    fn push_item(&mut self, value: T) -> usize {
        let (head, prev) = self.cur.map(|c| (c, self.items[c].prev)).unwrap_or((0, 0));

        let item = Item {
            val: Some(value),
            next: head,
            prev: prev,
        };

        let curr = if self.gaps.len() > 0 {
            let idx = self.gaps.pop().unwrap();
            self.items[idx] = item;
            idx
        } else {
            self.items.push(item);
            self.items.len() - 1
        };

        self.items[prev].next = curr;
        self.items[head].prev = curr;

        curr
    }

    /// Inserts a value at the right side of the queue, that is, at the tail.
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate ring_queue;
    /// # fn main() {
    /// use ring_queue::Ring;
    ///
    /// let mut r = ring![1, 2, 3];
    /// r.push(4);
    /// assert_eq!(r.collect_vec(), vec![1, 2, 3, 4]);
    /// # }
    /// ```
    pub fn push(&mut self, value: T) {
        let idx = self.push_item(value);
        self.cur = self.cur.or(Some(idx));
    }

    /// Inserts a value at the left side of the queue, that is, at the head.
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate ring_queue;
    /// # fn main() {
    /// use ring_queue::Ring;
    ///
    /// let mut r = ring![1, 2, 3];
    /// r.push_left(0);
    /// assert_eq!(r.collect_vec(), vec![0, 1, 2, 3]);
    /// # }
    /// ```
    pub fn push_left(&mut self, value: T) {
        let curr = self.push_item(value);
        self.cur = Some(curr);
    }

    #[inline]
    fn rotate_left(&mut self, n: isize) {
        self.cur = self.cur.map(|c| {
            let mut cur = c;
            for _ in n..0 {
                cur = self.items[cur].next;
            }
            cur
        });
    }

    #[inline]
    fn rotate_right(&mut self, n: isize) {
        self.cur = self.cur.map(|c| {
            let mut cur = c;
            for _ in 0..n {
                cur = self.items[cur].prev;
            }
            cur
        });
    }

    /// Rotate the deque `n` steps to the right. If `n` is negative, rotate to
    /// the left.
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate ring_queue;
    /// # fn main() {
    /// use ring_queue::Ring;
    ///
    /// let mut r = ring![1, 2, 3, 4, 5];
    ///
    /// r.rotate(1);
    /// assert_eq!(r.collect_vec(), vec![5, 1, 2, 3, 4]);
    ///
    /// r.rotate(-2);
    /// assert_eq!(r.collect_vec(), vec![2, 3, 4, 5, 1]);
    /// # }
    /// ```
    pub fn rotate(&mut self, n: isize) -> &mut Self {
        if n > 0 {
            self.rotate_right(n);
        } else if n < 0 {
            self.rotate_left(n);
        }
        self
    }

    #[inline]
    fn peek_left(&self, cur: usize, n: isize) -> Option<&T> {
        let mut c = cur;
        for _ in n..0 {
            c = self.items[c].prev;
        }
        self.items[c].val.as_ref()
    }

    #[inline]
    fn peek_right(&self, cur: usize, n: isize) -> Option<&T> {
        let mut c = cur;
        for _ in 0..n {
            c = self.items[c].next;
        }
        self.items[c].val.as_ref()
    }

    /// Retrieve the element `n` steps to the right from the head. If `n` is
    /// negative, the element `n` steps to the left from the head.
    /// This method can only be used if the element type implements the `Copy`
    /// trait.
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate ring_queue;
    /// # fn main() {
    /// use ring_queue::Ring;
    ///
    /// let r = ring![1, 2, 3, 4, 5];
    /// assert_eq!(r.peek(0), Some(&1));
    /// assert_eq!(r.peek(1), Some(&2));
    /// assert_eq!(r.peek(-1), Some(&5));
    /// # }
    /// ```
    pub fn peek(&self, n: isize) -> Option<&T> {
        match self.cur {
            Some(cur) => {
                if n < 0 {
                    self.peek_left(cur, n)
                } else {
                    self.peek_right(cur, n)
                }
            }
            None => None,
        }
    }

    /// Returns the length of the ring.
    pub fn len(&self) -> usize {
        self.items.len() - self.gaps.len()
    }

    /// Returns the current capacity of the ring.
    pub fn capacity(&self) -> usize {
        self.items.capacity()
    }

    /// Clears the ring, removing all values.
    /// Note that this method has no effect on its allocated capacity.
    ///
    /// # Example
    ///
    /// ```
    /// use ring_queue::Ring;
    ///
    /// let mut r = Ring::with_capacity(5);
    /// r.push(1);
    ///
    /// r.clear();
    /// assert_eq!(r.len(), 0);
    /// assert_eq!(r.capacity(), 5);
    /// ```
    pub fn clear(&mut self) {
        self.cur = None;
        self.gaps.clear();
        self.items.clear();
    }

    /// Moves all the elements of `other` into the right of `Self`, leaving
    /// `other` empty.
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate ring_queue;
    /// # fn main() {
    /// use ring_queue::Ring;
    ///
    /// let mut r = ring![1, 2, 3];
    /// r.append(&mut ring![4, 5, 6]);
    /// assert_eq!(r.collect_vec(), vec![1, 2, 3, 4, 5, 6]);
    /// # }
    /// ```
    pub fn append(&mut self, other: &mut Ring<T>) {
        for item in other.into_iter() {
            self.push(item);
        }
    }

    /// Moves all the elements of `other` into the left of `Self`, leaving
    /// `other` empty.
    /// Note that elements will be appended to the left in reverse with the
    /// same order they had in the other ring. That means they will be, in
    /// fact, in reverse order.
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate ring_queue;
    /// # fn main() {
    /// use ring_queue::Ring;
    ///
    /// let mut r = ring![4, 5, 6];
    /// r.append_left(&mut ring![3, 2, 1]);
    /// assert_eq!(r.collect_vec(), vec![1, 2, 3, 4, 5, 6]);
    /// # }
    /// ```
    pub fn append_left(&mut self, other: &mut Ring<T>) {
        for item in other.into_iter() {
            self.push_left(item);
        }
    }

    /// Reverses the elements in the ring.
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate ring_queue;
    /// # fn main() {
    /// use ring_queue::Ring;
    ///
    /// let mut r = ring![1, 2, 3];
    /// r.reverse();
    /// assert_eq!(r.collect_vec(), vec![3, 2, 1]);
    /// # }
    /// ```
    pub fn reverse(&mut self) {
        self.cur = self.cur.map(|cur| {
            let mut c = cur;
            let tail = self.items[c].prev;
            loop {
                let next = self.items[c].next;
                self.items[c].reverse();
                if next == cur {
                    break;
                }
                c = next;
            }
            tail
        });
    }

    /// Returns whether the ring is empty or not.
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate ring_queue;
    /// # fn main() {
    /// use ring_queue::Ring;
    ///
    /// assert_eq!(ring![1, 2, 3].is_empty(), false);
    /// assert_eq!(Ring::<i32>::new().is_empty(), true);
    /// # }
    /// ```
    pub fn is_empty(&self) -> bool {
        self.cur.is_none()
    }

    /// Return all the elements inside the ring as a vector. In order to use
    /// this method, the element type needs to implement the `Copy` trait.
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate ring_queue;
    /// # fn main() {
    /// use ring_queue::Ring;
    ///
    /// let r = ring![1, 2, 3];
    /// assert_eq!(r.collect_vec(), vec![1, 2, 3]);
    /// # }
    /// ```
    pub fn collect_vec(&self) -> Vec<T>
    where
        T: Copy,
    {
        let mut v = Vec::with_capacity(self.len());
        match self.cur {
            Some(cur) => {
                let mut c = cur;
                loop {
                    let item = self.items[c];
                    v.push(item.val.unwrap());
                    c = item.next;
                    if c == cur {
                        break;
                    }
                }
            }
            None => (),
        }
        v
    }
}

impl<'a, T> std::iter::IntoIterator for &'a Ring<T> where T: Copy {
    type Item = T;
    type IntoIter = Iter<'a, T>;

    /// Creates an iterator that will not consume the current ring. It will,
    /// instead, copy the elements one by one to the iterator.
    ///
    /// # Examples
    ///
    /// ```
    /// use ring_queue::Ring;
    ///
    /// let mut r = Ring::new();
    /// r.push(1);
    /// r.push(2);
    ///
    /// for i in r.into_iter() {
    ///     println!("{}", i);
    /// }
    ///
    /// // r is not empty now
    /// ```
    fn into_iter(self) -> Self::IntoIter {
        Iter {
            items: &self.items,
            head: self.cur.clone(),
            pos: self.cur.clone(),
        }
    }
}

impl<'a, T> std::iter::IntoIterator for &'a mut Ring<T> {
    type Item = T;
    type IntoIter = IterMut<T>;

    /// Creates a consuming iterator, that is, one that moves each value out of
    /// the ring (from head to tail). The ring will be empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use ring_queue::Ring;
    ///
    /// struct Foo { a: i32 }
    ///
    /// let mut r = Ring::new();
    /// r.push(Foo{a: 1});
    /// r.push(Foo{a: 2});
    ///
    /// for i in r.into_iter() {
    ///     println!("{}", i.a);
    /// }
    ///
    /// // r is empty now
    /// ```
    fn into_iter(self) -> Self::IntoIter {
        let mut items = Vec::new();
        let mut cur = None;

        std::mem::swap(&mut items, &mut self.items);
        std::mem::swap(&mut cur, &mut self.cur);
        self.items.clear();

        IterMut {
            items: items,
            head: cur,
            pos: cur,
        }
    }
}

impl<T> Clone for Ring<T>
where
    T: std::clone::Clone,
{
    fn clone(&self) -> Self {
        Ring {
            cur: self.cur,
            items: self.items.clone(),
            gaps: self.gaps.clone(),
        }
    }
}

pub struct Iter<'a, T: Copy> {
    items: &'a Vec<Item<T>>,
    head: Option<usize>,
    pos: Option<usize>,
}

impl<'a, T: Copy> Iterator for Iter<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        let mut val = None;
        self.pos = self.pos.and_then(|pos| {
            let item = self.items[pos];
            val = item.val;

            if item.next == self.head.unwrap() {
                None
            } else {
                Some(item.next)
            }
        });
        val
    }
}

pub struct IterMut<T> {
    items: Vec<Item<T>>,
    head: Option<usize>,
    pos: Option<usize>,
}

impl<T> Iterator for IterMut<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        let mut val = None;
        self.pos = self.pos.and_then(|pos| {
            let mut item = Item {
                val: None,
                next: 0,
                prev: 0,
            };
            std::mem::swap(&mut item, &mut self.items[pos]);
            val = item.val;

            if item.next == self.head.unwrap() {
                None
            } else {
                Some(item.next)
            }
        });
        val
    }
}

impl<T> std::iter::FromIterator<T> for Ring<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut ring = Ring::new();

        for item in iter {
            ring.push(item);
        }

        ring
    }
}

use std::fmt;

impl<T: fmt::Debug + Copy> fmt::Debug for Ring<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("[")?;
        match self.cur {
            Some(cur) => {
                let mut next = cur;

                loop {
                    f.write_fmt(format_args!("{:?}", self.items[next].val.unwrap()))?;
                    next = self.items[next].next;
                    if next == cur {
                        break;
                    } else {
                        f.write_str(", ")?;
                    }
                }
            }
            None => (),
        }
        f.write_str("]")
    }
}

impl<T> std::ops::Index<isize> for Ring<T> {
    type Output = T;

    fn index(&self, idx: isize) -> &T {
        match self.peek(idx) {
            Some(r) => r,
            None => panic!("tried to access index {} of empty ring", idx),
        }
    }
}

impl<T: PartialEq> PartialEq for Ring<T> {
    fn eq(&self, other: &Ring<T>) -> bool {
        if self.len() != other.len() {
            return false;
        }

        let cur = self.cur.unwrap();
        let cur2 = other.cur.unwrap();
        let mut next = cur;
        let mut next2 = cur2;

        loop {
            let v = &self.items[next].val;
            let v2 = &other.items[next2].val;
            if v != v2 {
                return false;
            }

            if next == cur && next2 == cur2 {
                break;
            } else if next == cur || next2 == cur2 {
                return false;
            }

            next = self.items[next].next;
            next2 = other.items[next2].next;
        }
        true
    }
}

impl<T: Eq> Eq for Ring<T> {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::iter::FromIterator;

    #[test]
    fn test_push() {
        let mut r = ring![1, 2];

        r.push(3);
        assert_eq!(r.collect_vec(), vec![1, 2, 3]);

        r.push_left(0);
        assert_eq!(r.collect_vec(), vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_debug() {
        let r = ring![1, 2, 3];
        assert_eq!("[1, 2, 3]", format!("{:?}", r));
    }

    #[test]
    fn test_iter_copyable() {
        let r = ring![1, 2, 3];
        let mut iter = r.into_iter();
        assert_eq!(iter.next(), Some(1));
        assert_eq!(iter.next(), Some(2));
        assert_eq!(iter.next(), Some(3));
        assert_eq!(iter.next(), None);

        assert_eq!(r.len(), 3);
    }

    #[test]
    fn test_iter_nocopyable() {
        let mut r = ring![vec![1, 2], vec![3, 4]];
        let mut iter = r.into_iter();
        assert_eq!(iter.next(), Some(vec![1, 2]));
        assert_eq!(iter.next(), Some(vec![3, 4]));
        assert_eq!(iter.next(), None);

        assert_eq!(r.is_empty(), true);
    }

    #[test]
    fn test_collect() {
        let r = ring![1, 2, 3];
        assert_eq!(vec![1, 2, 3], r.collect_vec());
    }

    #[test]
    fn test_from_iter() {
        let r = Ring::from_iter(vec![1, 2, 3]);
        assert_eq!(vec![1, 2, 3], r.collect_vec());
    }

    #[test]
    fn test_peek() {
        let r = ring![1, 2, 3, 4];
        assert_eq!(r.peek(0), Some(&1));
        assert_eq!(r.peek(1), Some(&2));
        assert_eq!(r.peek(4), Some(&1));
        assert_eq!(r.peek(-1), Some(&4));
        assert_eq!(r.peek(-1), Some(&4));
        assert_eq!(r.peek(-6), Some(&3));
    }

    #[test]
    fn test_rotate() {
        let mut r1 = ring![1, 2, 3, 4];
        assert_eq!(r1.rotate(1).collect_vec(), vec![4, 1, 2, 3]);

        let mut r2 = ring![1, 2, 3, 4];
        assert_eq!(r2.rotate(0).collect_vec(), vec![1, 2, 3, 4]);

        let mut r3 = ring![1, 2, 3, 4];
        assert_eq!(r3.rotate(-1).collect_vec(), vec![2, 3, 4, 1]);
    }

    #[test]
    fn test_pop() {
        let mut r = ring![1, 2, 3, 4];

        assert_eq!(r.pop(), Some(4));
        assert_eq!(r.pop_left(), Some(1));
        assert_eq!(r.collect_vec(), vec![2, 3]);
    }

    #[test]
    fn test_len() {
        let r = ring![1, 2, 3, 4];
        assert_eq!(r.len(), 4);
    }

    #[test]
    fn test_capacity() {
        let r: Ring<i32> = Ring::with_capacity(6);
        assert_eq!(r.capacity(), 6);
        assert_eq!(r.len(), 0);
    }

    #[test]
    fn test_clear() {
        let mut r = Ring::with_capacity(5);
        r.push(1);
        r.push(2);
        r.push(3);

        assert_eq!(r.len(), 3);
        assert_eq!(r.capacity(), 5);

        r.clear();
        assert_eq!(r.len(), 0);
        assert_eq!(r.capacity(), 5);
    }

    #[test]
    fn test_push_with_gaps() {
        let mut r = ring![1, 2, 3, 4, 5];
        assert_eq!(r.rotate(-1).pop_left(), Some(2));
        assert_eq!(r.gaps.len(), 1);

        r.push(6);
        assert_eq!(r.gaps.len(), 0);
        assert_eq!(r.items[1].val, Some(6));
        r.push_left(7);

        assert_eq!(r.collect_vec(), vec![7, 3, 4, 5, 1, 6]);
    }

    #[test]
    fn test_append() {
        let mut r = ring![1, 2, 3];
        r.append(&mut ring![4, 5, 6]);
        assert_eq!(r.collect_vec(), vec![1, 2, 3, 4, 5, 6]);

        r.append_left(&mut ring![7, 8, 9]);
        assert_eq!(r.collect_vec(), vec![9, 8, 7, 1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_reverse() {
        let mut r = ring![1, 2, 3];
        r.reverse();
        assert_eq!(r.collect_vec(), vec![3, 2, 1]);
    }

    #[test]
    fn test_is_empty() {
        let mut r = Ring::new();
        assert_eq!(r.is_empty(), true);

        r.push(1);
        assert_eq!(r.is_empty(), false);

        r.clear();
        assert_eq!(r.is_empty(), true);
    }

    #[test]
    fn test_macro_nocopy() {
        let r = ring![vec![1, 2, 3]];
        assert_eq!(r.is_empty(), false);
    }

    #[test]
    fn test_eq() {
        let r = ring![1, 2, 3];

        assert_eq!(r == ring![1, 2], false);
        assert_eq!(r == ring![1, 2, 3], true);
        assert_eq!(r == ring![], false);

        assert_eq!(r != ring![1, 2], true);
        assert_eq!(r != ring![1, 2, 3], false);
        assert_eq!(r != ring![], true);
    }

    #[test]
    fn test_index() {
        let r = ring![1, 2, 3];
        assert_eq!(r[0], 1);
        assert_eq!(r[1], 2);
        assert_eq!(r[-1], 3);
    }
}
