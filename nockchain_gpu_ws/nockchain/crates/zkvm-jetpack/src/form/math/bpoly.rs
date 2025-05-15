use std::vec;

use crate::form::math::{bpow, FieldError};
use crate::form::poly::*;

pub fn bpadd(a: &[Belt], b: &[Belt], res: &mut [Belt]) {
    let min: &[Belt];
    let max: &[Belt];
    if a.len() <= b.len() {
        min = a;
        max = b;
    } else {
        min = b;
        max = a;
    }

    for ((res_vec, max_vec), min_vec) in res
        .iter_mut()
        .zip(max)
        .zip(min.iter().map(Some).chain(std::iter::repeat(None)))
    {
        if let Some(min_vec) = min_vec {
            *res_vec = *min_vec + *max_vec;
        } else {
            *res_vec = *max_vec;
        }
    }
}

#[inline(always)]
pub fn bpadd_(left: &[Belt], right: &[Belt]) -> Vec<Belt> {
    let len = std::cmp::max(left.len(), right.len());
    let mut res = vec![Belt::zero(); len];
    bpadd(left, right, res.as_mut_slice());
    res
}

#[inline(always)]
pub fn bpadd_in_place(a: &mut [Belt], b: &[Belt]) {
    assert!(a.len() >= b.len());

    for (a_belt, b_belt) in a
        .iter_mut()
        .zip(b.iter().map(Some).chain(std::iter::repeat(None)))
    {
        if let Some(b_belt) = b_belt {
            *a_belt = *b_belt + *a_belt;
        } else {
            break;
        }
    }
}

#[inline(always)]
pub fn bpsub(a: &[Belt], b: &[Belt], res: &mut [Belt]) {
    let a_len = a.len();
    let b_len = b.len();

    let res_len = std::cmp::max(a_len, b_len);

    for i in 0..res_len {
        let n = i;
        if i < a_len && i < b_len {
            res[n] = a[n] - b[n];
        } else if i < a_len {
            res[n] = a[n];
        } else {
            res[n] = -b[n];
        }
    }
}

#[inline(always)]
pub fn bpsub_in_place(a: &mut [Belt], b: &[Belt]) {
    debug_assert!(a.len() >= b.len());
    for (max_vec, min_vec) in a
        .iter_mut()
        .zip(b.iter().map(Some).chain(std::iter::repeat(None)))
    {
        if let Some(min_vec) = min_vec {
            *max_vec = *max_vec - *min_vec;
        } else {
            break;
        }
    }
    //  TODO: hoon impl does not normalize here, but maybe it should?
    //normalize_poly(a)
}

#[inline(always)]
pub fn bpsub_(left: &[Belt], right: &[Belt]) -> Vec<Belt> {
    let len = std::cmp::max(left.len(), right.len());
    let mut res = vec![Belt::zero(); len];
    bpsub(left, right, res.as_mut_slice());
    res
}

#[inline(always)]
pub fn bpmul(a: &[Belt], b: &[Belt], res: &mut [Belt]) {
    if a.is_zero() || b.is_zero() {
        res.fill(Belt(0));
        return;
    }

    res.fill(Belt(0));

    let a_len = a.len();
    let b_len = b.len();

    for i in 0..a_len {
        if a[i] == 0 {
            continue;
        }
        for j in 0..b_len {
            res[i + j] = res[i + j] + a[i] * b[j];
        }
    }
}

#[inline(always)]
pub fn bpmul_(left: &[Belt], right: &[Belt]) -> Vec<Belt> {
    let len = left.len() + right.len() - 1;
    let mut res = vec![Belt::zero(); len];
    bpmul(left, right, res.as_mut_slice());
    res
}

#[inline(always)]
pub fn bpscal(scalar: Belt, b: &[Belt], res: &mut [Belt]) {
    for (res, bp) in res.iter_mut().zip(b.iter()) {
        *res = scalar * *bp;
    }
}

#[inline(always)]
pub fn bpscal_(scalar: Belt, b: &[Belt]) -> Vec<Belt> {
    let mut res = vec![Belt(0); b.len()];
    bpscal(scalar, b, res.as_mut_slice());
    res
}

#[inline(always)]
pub fn bp_hadamard(a: &[Belt], b: &[Belt], res: &mut [Belt]) {
    assert_eq!(
        a.len(),
        b.len(),
        "Unequal lengths: {}, {}",
        a.len(),
        b.len()
    );
    res.iter_mut()
        .zip(a.iter())
        .zip(b.iter())
        .for_each(|((res_i, a_i), b_i)| {
            *res_i = *a_i * *b_i;
        });
}

#[inline(always)]
pub fn bp_hadamard_(a: &[Belt], b: &[Belt]) -> Vec<Belt> {
    assert_eq!(
        a.len(),
        b.len(),
        "Unequal lengths: {}, {}",
        a.len(),
        b.len()
    );
    let mut res = vec![Belt(0); a.len()];
    res.iter_mut()
        .zip(a.iter())
        .zip(b.iter())
        .for_each(|((res_i, a_i), b_i)| {
            *res_i = *a_i * *b_i;
        });
    res
}

#[inline(always)]
pub fn bpneg(b: &[Belt], res: &mut [Belt]) {
    for (res, bp) in res.iter_mut().zip(b.iter()) {
        *res = -*bp;
    }
}

#[inline(always)]
pub fn bppow(a: &[Belt], mut n: usize) -> Vec<Belt> {
    let mut q = vec![Belt(1)];
    let mut p = a.to_vec();
    while n != 0 {
        if n & 1 == 1 {
            q = bpmul_(&q, &p);
        } else {
            p = bpmul_(&p, &p);
        }
        n >>= 1;
    }
    q
}

#[inline]
fn bitreverse(mut n: u32, l: u32) -> u32 {
    let mut r = 0;
    for _ in 0..l {
        r = (r << 1) | (n & 1);
        n >>= 1;
    }
    r
}

#[inline(always)]
pub fn bp_fft(bp: &[Belt]) -> Result<Vec<Belt>, FieldError> {
    let order: Belt = Belt(bp.len() as u64);
    let root = order.ordered_root()?;
    Ok(bp_ntt(bp, &root))
}

pub fn bp_ntt(bp: &[Belt], root: &Belt) -> Vec<Belt> {
    let n = bp.len() as u32;

    if n == 1 {
        return vec![bp[0]];
    }

    debug_assert!(n.is_power_of_two());

    let log_2_of_n = n.ilog2();

    let mut x: Vec<Belt> = vec![Belt(0); n as usize];
    x.copy_from_slice(bp);

    for k in 0..n {
        let rk = bitreverse(k, log_2_of_n);
        if k < rk {
            x.swap(rk as usize, k as usize);
        }
    }

    let mut m = 1;
    for _ in 0..log_2_of_n {
        let w_m: Belt = bpow(root.0, (n / (2 * m)) as u64).into();

        let mut k = 0;
        while k < n {
            let mut w = Belt(1);

            for j in 0..m {
                let u: Belt = x[(k + j) as usize];
                let v: Belt = x[(k + j + m) as usize] * w;
                x[(k + j) as usize] = u + v;
                x[(k + j + m) as usize] = u - v;
                w = w * w_m;
            }

            k += 2 * m;
        }

        m *= 2;
    }
    x
}

#[inline(always)]
pub fn bp_shift(poly_a: &[Belt], belt_b: &Belt, poly_res: &mut [Belt]) {
    let mut belt_power: Belt = Belt(1);

    for i in 0..poly_a.len() {
        poly_res[i] = poly_a[i] * belt_power;
        belt_power = belt_power * *belt_b;
    }
}

#[inline(always)]
pub fn bp_coseword(bp: &[Belt], offset: &Belt, order: u32, root: &Belt) -> Vec<Belt> {
    // shift
    let len_res: u32 = order;
    let mut res = vec![Belt::zero(); len_res as usize];
    bp_shift(bp, offset, &mut res);

    bp_ntt(&res, root)
}

#[inline(always)]
pub fn bpoly_zero_extend(a: &[Belt], res: &mut [Belt]) {
    let a_len = a.len();
    let res_len = res.len();
    res[0..a_len].copy_from_slice(a);
    res[a_len..res_len].fill(Belt::zero());
}

#[inline(always)]
pub fn normalize_bpoly(a: &mut Vec<Belt>) {
    // normalize result by removing trailing zeros
    if a.len() <= 1 {
        return;
    }
    for i in (0..a.len()).rev() {
        if a[i].is_zero() {
            a.pop();
        } else {
            break;
        }
    }
}
