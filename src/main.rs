//! Ultimate-Performance X25519 vanity key finder in pure Rust from scratch.
//!
//! ## Architecture & Optimization Strategy:
//! 1. **Twisted Edwards Extended Coordinates**: (X:Y:Z:T) with a = -1.
//! 2. **Precomputed Radix-64 (w = 6) Table**: Used for initial random base point scalar mult.
//! 3. **Incremental Point Stepping (Q_{j+1} = Q_j + 8*B)**:
//!    Instead of running a full 43-addition scalar mult for every key, each candidate key is
//!    obtained via a single mixed point addition with 8*B (in Niels form), reducing the cost
//!    from 43 point additions down to **1 point addition (7 field multiplications)** per key!
//! 4. **Montgomery Batch Inversion**: Inverts candidate denominators simultaneously (e.g. 256 keys),
//!    amortizing the 255-bit Fermat inversion chain to ~3 field multiplications per key.
//! 5. **Direct Birational Mapping without Z Inversion**:
//!    u = (Z + Y) / (Z - Y), eliminating the need to invert Z.
//! 6. **Zero-Allocation Direct Byte Filter**: Pre-filters public key byte 0 against the target
//!    prefix in O(1) before any Base64 encoding.
//! 7. **Cryptographically Secure Randomness**: Private-key seeds come from the operating system RNG.

#![allow(clippy::many_single_char_names)]

use base64::{engine::general_purpose::STANDARD as B64, Engine as _};
use rand::{rngs::OsRng, RngCore};
use std::sync::atomic::{AtomicI32, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

// ══════════════════════════════════════════════════════════════════════════════
// GF(2^255 - 19) Radix 2^51 Field Element
// ══════════════════════════════════════════════════════════════════════════════

const MASK51: u64 = (1u64 << 51) - 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fe(pub [u64; 5]);

pub const ZERO: Fe = Fe([0, 0, 0, 0, 0]);
pub const ONE: Fe = Fe([1, 0, 0, 0, 0]);

// 2*d where d = -121665/121666 (mod p)
pub const D2: Fe = Fe([
    1859910466990425,
    932731440258426,
    1072319116312658,
    1815898335770999,
    633789495995903,
]);

impl Fe {
    #[inline(always)]
    pub fn add(self, b: Fe) -> Fe {
        Fe([
            self.0[0] + b.0[0],
            self.0[1] + b.0[1],
            self.0[2] + b.0[2],
            self.0[3] + b.0[3],
            self.0[4] + b.0[4],
        ])
    }

    #[inline(always)]
    pub fn sub(self, b: Fe) -> Fe {
        let a = self.0;
        let b = b.0;
        Fe([
            a[0] + 9007199254740916 - b[0],
            a[1] + 9007199254740988 - b[1],
            a[2] + 9007199254740988 - b[2],
            a[3] + 9007199254740988 - b[3],
            a[4] + 9007199254740988 - b[4],
        ])
    }

    #[inline(always)]
    pub fn mul(self, b: Fe) -> Fe {
        let [a0, a1, a2, a3, a4] = self.0;
        let [b0, b1, b2, b3, b4] = b.0;

        let b1_19 = b1 * 19;
        let b2_19 = b2 * 19;
        let b3_19 = b3 * 19;
        let b4_19 = b4 * 19;

        let mut c0 = (a0 as u128) * (b0 as u128)
            + (a1 as u128) * (b4_19 as u128)
            + (a2 as u128) * (b3_19 as u128)
            + (a3 as u128) * (b2_19 as u128)
            + (a4 as u128) * (b1_19 as u128);
        let mut c1 = (a0 as u128) * (b1 as u128)
            + (a1 as u128) * (b0 as u128)
            + (a2 as u128) * (b4_19 as u128)
            + (a3 as u128) * (b3_19 as u128)
            + (a4 as u128) * (b2_19 as u128);
        let mut c2 = (a0 as u128) * (b2 as u128)
            + (a1 as u128) * (b1 as u128)
            + (a2 as u128) * (b0 as u128)
            + (a3 as u128) * (b4_19 as u128)
            + (a4 as u128) * (b3_19 as u128);
        let mut c3 = (a0 as u128) * (b3 as u128)
            + (a1 as u128) * (b2 as u128)
            + (a2 as u128) * (b1 as u128)
            + (a3 as u128) * (b0 as u128)
            + (a4 as u128) * (b4_19 as u128);
        let mut c4 = (a0 as u128) * (b4 as u128)
            + (a1 as u128) * (b3 as u128)
            + (a2 as u128) * (b2 as u128)
            + (a3 as u128) * (b1 as u128)
            + (a4 as u128) * (b0 as u128);

        macro_rules! carry {
            ($lo:expr, $hi:expr) => {{
                let c = $lo >> 51;
                $lo &= MASK51 as u128;
                $hi += c;
            }};
        }
        carry!(c0, c1);
        carry!(c1, c2);
        carry!(c2, c3);
        carry!(c3, c4);
        let top = c4 >> 51;
        c4 &= MASK51 as u128;
        c0 += top * 19;
        carry!(c0, c1);

        Fe([c0 as u64, c1 as u64, c2 as u64, c3 as u64, c4 as u64])
    }

    #[inline(always)]
    pub fn sq(self) -> Fe {
        let [a0, a1, a2, a3, a4] = self.0;
        let a0d = a0 * 2;
        let a1d = a1 * 2;
        let a2d = a2 * 2;
        let a3d = a3 * 2;
        let a4_19 = a4 * 19;
        let a3_19 = a3 * 19;

        let mut c0 = (a0 as u128) * (a0 as u128)
            + (a1d as u128) * (a4_19 as u128)
            + (a2d as u128) * (a3_19 as u128);
        let mut c1 = (a0d as u128) * (a1 as u128)
            + (a2d as u128) * (a4_19 as u128)
            + (a3 as u128) * (a3_19 as u128);
        let mut c2 = (a0d as u128) * (a2 as u128)
            + (a1 as u128) * (a1 as u128)
            + (a3d as u128) * (a4_19 as u128);
        let mut c3 = (a0d as u128) * (a3 as u128)
            + (a1d as u128) * (a2 as u128)
            + (a4 as u128) * (a4_19 as u128);
        let mut c4 = (a0d as u128) * (a4 as u128)
            + (a1d as u128) * (a3 as u128)
            + (a2 as u128) * (a2 as u128);

        macro_rules! carry {
            ($lo:expr, $hi:expr) => {{
                let c = $lo >> 51;
                $lo &= MASK51 as u128;
                $hi += c;
            }};
        }
        carry!(c0, c1);
        carry!(c1, c2);
        carry!(c2, c3);
        carry!(c3, c4);
        let top = c4 >> 51;
        c4 &= MASK51 as u128;
        c0 += top * 19;
        carry!(c0, c1);

        Fe([c0 as u64, c1 as u64, c2 as u64, c3 as u64, c4 as u64])
    }

    #[inline(always)]
    pub fn neg(self) -> Fe {
        Fe([
            9007199254740916 - self.0[0],
            9007199254740988 - self.0[1],
            9007199254740988 - self.0[2],
            9007199254740988 - self.0[3],
            9007199254740988 - self.0[4],
        ])
    }

    #[inline(always)]
    pub fn mul_small(self, s: u64) -> Fe {
        let mut c: [u128; 5] = [
            self.0[0] as u128 * s as u128,
            self.0[1] as u128 * s as u128,
            self.0[2] as u128 * s as u128,
            self.0[3] as u128 * s as u128,
            self.0[4] as u128 * s as u128,
        ];
        macro_rules! carry {
            ($lo:expr, $hi:expr) => {{
                let cv = $lo >> 51;
                $lo &= MASK51 as u128;
                $hi += cv;
            }};
        }
        carry!(c[0], c[1]);
        carry!(c[1], c[2]);
        carry!(c[2], c[3]);
        carry!(c[3], c[4]);
        let top = c[4] >> 51;
        c[4] &= MASK51 as u128;
        c[0] += top * 19;
        carry!(c[0], c[1]);
        Fe([
            c[0] as u64, c[1] as u64, c[2] as u64, c[3] as u64, c[4] as u64,
        ])
    }

    pub fn invert(self) -> Fe {
        let a = self;
        let a2 = a.sq();
        let a4 = a2.sq();
        let a8 = a4.sq();
        let a9 = a8.mul(a);
        let a11 = a9.mul(a2);
        let a22 = a11.sq();
        let v5 = a22.mul(a9);
        let v10 = sq_n_mul(v5, 5, v5);
        let v20 = sq_n_mul(v10, 10, v10);
        let v40 = sq_n_mul(v20, 20, v20);
        let v50 = sq_n_mul(v40, 10, v10);
        let v100 = sq_n_mul(v50, 50, v50);
        let v200 = sq_n_mul(v100, 100, v100);
        let v250 = sq_n_mul(v200, 50, v50);
        sq_n_mul(v250, 5, a11)
    }

    /// Fast extraction of byte 0 of the canonical encoded field element.
    /// Performs only the reduction needed for limb 0, without full 32-byte serialization.
    #[inline(always)]
    pub fn encode_byte0(self) -> u8 {
        let mut h = self.0;
        let c0 = h[0] >> 51;
        h[0] &= MASK51;
        h[1] += c0;
        let c1 = h[1] >> 51;
        h[1] &= MASK51;
        h[2] += c1;
        let c2 = h[2] >> 51;
        h[2] &= MASK51;
        h[3] += c2;
        let c3 = h[3] >> 51;
        h[3] &= MASK51;
        h[4] += c3;
        let top = h[4] >> 51;
        h[4] &= MASK51;
        h[0] += top * 19;
        let c0_final = h[0] >> 51;
        h[0] &= MASK51;
        h[1] += c0_final;

        let (b0, bor0) = borrowing_sub(h[0], MASK51 - 18, 0);
        let (_, bor1) = borrowing_sub(h[1], MASK51, bor0);
        let (_, bor2) = borrowing_sub(h[2], MASK51, bor1);
        let (_, bor3) = borrowing_sub(h[3], MASK51, bor2);
        let (_, bor4) = borrowing_sub(h[4], MASK51, bor3);

        let keep_h = (bor4 as u64).wrapping_neg();
        let h0 = (h[0] & keep_h) | (b0 & !keep_h);
        h0 as u8
    }


    pub fn encode(self) -> [u8; 32] {
        let mut h = self.0;
        macro_rules! carry {
            ($i:expr, $j:expr) => {{
                let c = h[$i] >> 51;
                h[$i] &= MASK51;
                h[$j] += c;
            }};
        }
        carry!(0, 1);
        carry!(1, 2);
        carry!(2, 3);
        carry!(3, 4);
        let top = h[4] >> 51;
        h[4] &= MASK51;
        h[0] += top * 19;
        carry!(0, 1);

        let (b0, bor0) = borrowing_sub(h[0], MASK51 - 18, 0);
        let (b1, bor1) = borrowing_sub(h[1], MASK51, bor0);
        let (b2, bor2) = borrowing_sub(h[2], MASK51, bor1);
        let (b3, bor3) = borrowing_sub(h[3], MASK51, bor2);
        let (b4, bor4) = borrowing_sub(h[4], MASK51, bor3);
        let g = [b0, b1, b2, b3, b4];

        let keep_h = (bor4 as u64).wrapping_neg();
        let [h0, h1, h2, h3, h4] = [
            (h[0] & keep_h) | (g[0] & !keep_h),
            (h[1] & keep_h) | (g[1] & !keep_h),
            (h[2] & keep_h) | (g[2] & !keep_h),
            (h[3] & keep_h) | (g[3] & !keep_h),
            (h[4] & keep_h) | (g[4] & !keep_h),
        ];

        let mut out = [0u8; 32];
        out[0] = h0 as u8;
        out[1] = (h0 >> 8) as u8;
        out[2] = (h0 >> 16) as u8;
        out[3] = (h0 >> 24) as u8;
        out[4] = (h0 >> 32) as u8;
        out[5] = (h0 >> 40) as u8;
        out[6] = ((h0 >> 48) | (h1 << 3)) as u8;
        out[7] = (h1 >> 5) as u8;
        out[8] = (h1 >> 13) as u8;
        out[9] = (h1 >> 21) as u8;
        out[10] = (h1 >> 29) as u8;
        out[11] = (h1 >> 37) as u8;
        out[12] = ((h1 >> 45) | (h2 << 6)) as u8;
        out[13] = (h2 >> 2) as u8;
        out[14] = (h2 >> 10) as u8;
        out[15] = (h2 >> 18) as u8;
        out[16] = (h2 >> 26) as u8;
        out[17] = (h2 >> 34) as u8;
        out[18] = (h2 >> 42) as u8;
        out[19] = ((h2 >> 50) | (h3 << 1)) as u8;
        out[20] = (h3 >> 7) as u8;
        out[21] = (h3 >> 15) as u8;
        out[22] = (h3 >> 23) as u8;
        out[23] = (h3 >> 31) as u8;
        out[24] = (h3 >> 39) as u8;
        out[25] = ((h3 >> 47) | (h4 << 4)) as u8;
        out[26] = (h4 >> 4) as u8;
        out[27] = (h4 >> 12) as u8;
        out[28] = (h4 >> 20) as u8;
        out[29] = (h4 >> 28) as u8;
        out[30] = (h4 >> 36) as u8;
        out[31] = (h4 >> 44) as u8;
        out
    }

    pub fn decode(b: &[u8; 32]) -> Fe {
        let load64 = |i: usize| u64::from_le_bytes(b[i..i + 8].try_into().unwrap());
        Fe([
            load64(0) & MASK51,
            (load64(6) >> 3) & MASK51,
            (load64(12) >> 6) & MASK51,
            (load64(19) >> 1) & MASK51,
            (load64(24) >> 12) & MASK51,
        ])
    }
}

#[inline(always)]
fn borrowing_sub(a: u64, b: u64, borrow: u64) -> (u64, u64) {
    let (r, ov1) = a.overflowing_sub(b);
    let (r, ov2) = r.overflowing_sub(borrow);
    (r & MASK51, (ov1 | ov2) as u64)
}

#[inline]
fn sq_n_mul(mut x: Fe, n: usize, m: Fe) -> Fe {
    for _ in 0..n {
        x = x.sq();
    }
    x.mul(m)
}

// ══════════════════════════════════════════════════════════════════════════════
// Extended Twisted Edwards Coordinates (X : Y : Z : T) & Niels Form
// ══════════════════════════════════════════════════════════════════════════════

#[derive(Clone, Copy)]
pub struct ExtPoint {
    pub x: Fe,
    pub y: Fe,
    pub z: Fe,
    pub t: Fe,
}

#[derive(Clone, Copy)]
pub struct NielsPoint {
    pub ypx: Fe,
    pub ymx: Fe,
    pub td2: Fe,
}

impl ExtPoint {
    pub fn identity() -> Self {
        ExtPoint {
            x: ZERO,
            y: ONE,
            z: ONE,
            t: ZERO,
        }
    }

    /// EFD dbl-2008-hwcd with a = -1 (Twisted Edwards):
    /// Cost: 4M + 4S
    #[inline(always)]
    pub fn double(&self) -> ExtPoint {
        let a = self.x.sq();
        let b = self.y.sq();
        let c = self.z.sq().mul_small(2);
        let e = self.x.add(self.y).sq().sub(a).sub(b);
        let g = b.sub(a);
        let f = g.sub(c);
        let h = a.add(b).neg(); // -(a+b)
        ExtPoint {
            x: e.mul(f),
            y: g.mul(h),
            z: f.mul(g),
            t: e.mul(h),
        }
    }

    /// EFD madd-2008-hwcd:
    /// Cost: 7 field multiplications + 1 mul_small
    #[inline(always)]
    pub fn add_niels(&self, q: &NielsPoint) -> ExtPoint {
        let a = self.y.sub(self.x).mul(q.ymx);
        let b = self.y.add(self.x).mul(q.ypx);
        let c = self.t.mul(q.td2);
        let d = self.z.mul_small(2);
        let e = b.sub(a);
        let f = d.sub(c);
        let g = d.add(c);
        let h = b.add(a);
        ExtPoint {
            x: e.mul(f),
            y: g.mul(h),
            z: f.mul(g),
            t: e.mul(h),
        }
    }
}

pub fn base_point() -> ExtPoint {
    let y_bytes: [u8; 32] = [
        0x58, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66,
        0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66,
        0x66, 0x66,
    ];
    let x_bytes: [u8; 32] = [
        0x1a, 0xd5, 0x25, 0x8f, 0x60, 0x2d, 0x56, 0xc9, 0xb2, 0xa7, 0x25, 0x95, 0x60, 0xc7, 0x2c,
        0x69, 0x5c, 0xdc, 0xd6, 0xfd, 0x31, 0xe2, 0xa4, 0xc0, 0xfe, 0x53, 0x6e, 0xcd, 0xd3, 0x36,
        0x69, 0x21,
    ];
    let x = Fe::decode(&x_bytes);
    let y = Fe::decode(&y_bytes);
    ExtPoint {
        x,
        y,
        z: ONE,
        t: x.mul(y),
    }
}

pub fn affine_to_niels(x: Fe, y: Fe) -> NielsPoint {
    NielsPoint {
        ypx: y.add(x),
        ymx: y.sub(x),
        td2: x.mul(y).mul(D2),
    }
}

pub fn point_to_niels(p: &ExtPoint) -> NielsPoint {
    let zi = p.z.invert();
    affine_to_niels(p.x.mul(zi), p.y.mul(zi))
}

// ══════════════════════════════════════════════════════════════════════════════
// Radix-64 (w = 6) Precomputed Base-Point Table: 43 windows × 64 entries
// ══════════════════════════════════════════════════════════════════════════════

pub const WINDOWS_6: usize = 43;
pub const WIN_SIZE_6: usize = 64;

pub struct TableW6 {
    pub windows: [[NielsPoint; WIN_SIZE_6]; WINDOWS_6],
    pub step_8b: NielsPoint, // Precomputed 8 * B in Niels form (for clamped-key stepping)
    pub step_1b: NielsPoint, // Precomputed 1 * B in Niels form (for raw scalar stepping)
}

pub fn build_table_w6() -> Box<TableW6> {
    let id_niels = NielsPoint {
        ypx: ONE,
        ymx: ONE,
        td2: ZERO,
    };
    let mut windows = [[id_niels; WIN_SIZE_6]; WINDOWS_6];
    let mut block_base = base_point();

    for i in 0..WINDOWS_6 {
        let mut acc = ExtPoint::identity();
        let base_niels = point_to_niels(&block_base);
        for j in 1..WIN_SIZE_6 {
            acc = acc.add_niels(&base_niels);
            windows[i][j] = point_to_niels(&acc);
        }
        if i + 1 < WINDOWS_6 {
            for _ in 0..6 {
                block_base = block_base.double();
            }
        }
    }

    // 1 * B = base point itself, used for Q ← Q + B incremental stepping
    let step_1b = point_to_niels(&base_point());

    // 8 * B = double(double(double(B))), used for clamped-key stepping
    let mut b8 = base_point();
    b8 = b8.double().double().double();
    let step_8b = point_to_niels(&b8);

    Box::new(TableW6 { windows, step_8b, step_1b })
}

#[inline(always)]
fn get_chunk6(k: &[u8; 32], i: usize) -> usize {
    let bit_start = 6 * i;
    let byte_idx = bit_start >> 3;
    let bit_offset = bit_start & 7;
    let b1 = if byte_idx + 1 < 32 {
        k[byte_idx + 1] as u32
    } else {
        0
    };
    let val = k[byte_idx] as u32 | (b1 << 8);
    ((val >> bit_offset) & 0x3f) as usize
}

#[inline(always)]
pub fn scalar_mult_w6(tbl: &TableW6, k: &[u8; 32]) -> ExtPoint {
    let mut acc = ExtPoint::identity();
    for i in 0..WINDOWS_6 {
        acc = acc.add_niels(&tbl.windows[i][get_chunk6(k, i)]);
    }
    acc
}

pub fn x25519_single(tbl: &TableW6, priv_key: &[u8; 32]) -> [u8; 32] {
    let mut k = *priv_key;
    k[0] &= 248;
    k[31] &= 127;
    k[31] |= 64;

    let q = scalar_mult_w6(tbl, &k);
    let num = q.z.add(q.y);
    let den = q.z.sub(q.y);
    num.mul(den.invert()).encode()
}

// ══════════════════════════════════════════════════════════════════════════════
// Incremental Stepping Batch Inversion Engine (1024 Keys Batch)
// ══════════════════════════════════════════════════════════════════════════════

/// Number of keys produced per batch-inversion call.
/// Larger = one Fe::invert() amortized over more keys (diminishing returns past 512).
pub const STEP_BATCH: usize = 1024;

/// Performs STEP_BATCH consecutive incremental steps from (cur_q, cur_priv)
/// with only 1 point addition per step and 1 simultaneous Montgomery batch inversion!
#[inline(always)]
pub fn step_batch_256(
    tbl: &TableW6,
    cur_q: &mut ExtPoint,
    cur_priv: &mut [u8; 32],
    pubs: &mut [[u8; 32]; STEP_BATCH],
    privs: &mut [[u8; 32]; STEP_BATCH],
) {
    let mut nums = [ZERO; STEP_BATCH];
    let mut dens = [ZERO; STEP_BATCH];

    for i in 0..STEP_BATCH {
        *cur_q = cur_q.add_niels(&tbl.step_8b);
        // Add 8 to 256-bit little-endian integer in cur_priv
        let mut carry = 8u64;
        for b in cur_priv.iter_mut() {
            let sum = *b as u64 + carry;
            *b = sum as u8;
            carry = sum >> 8;
            if carry == 0 {
                break;
            }
        }
        privs[i] = *cur_priv;
        nums[i] = cur_q.z.add(cur_q.y);
        dens[i] = cur_q.z.sub(cur_q.y);
    }

    // Montgomery Batch Inversion on 256 elements
    let mut prefix_prods = [ZERO; STEP_BATCH];
    prefix_prods[0] = dens[0];
    for i in 1..STEP_BATCH {
        prefix_prods[i] = prefix_prods[i - 1].mul(dens[i]);
    }

    let mut inv_all = prefix_prods[STEP_BATCH - 1].invert();

    let mut den_invs = [ZERO; STEP_BATCH];
    for i in (1..STEP_BATCH).rev() {
        den_invs[i] = inv_all.mul(prefix_prods[i - 1]);
        inv_all = inv_all.mul(dens[i]);
    }
    den_invs[0] = inv_all;

    for i in 0..STEP_BATCH {
        pubs[i] = nums[i].mul(den_invs[i]).encode();
    }
}

/// Performs 256 consecutive incremental steps Q ← Q + B (step size = 1).
///
/// This is the **raw scalar** variant: `cur_raw` is a 256-bit little-endian
/// counter that is incremented by 1 each step (no clamping during stepping).
/// The elliptic curve point Q tracks `cur_raw * B` exactly.
///
/// At output time the raw counter is X25519-clamped:
///   `priv_clamped = cur_raw  with  bits[0..2] = 0, bit[254] = 1, bit[255] = 0`
///
/// Because clamping only masks/forces 4 bits the clamped private key is
/// always a valid X25519 scalar and the stored public key is:
///   `pub = (cur_raw * B) converted to Montgomery u` (birational map)
///
/// **Note**: consecutive raw scalars with the same top bits differ only in
/// the low-3 bits, so their clamped forms are identical – eight consecutive
/// steps share one effective public key.  Use this function when you want
/// maximum raw iteration speed for scanning unclamped key-space; use
/// `step_batch_256` (which steps by 8*B) to enumerate only distinct
/// clamped public keys.
#[inline(always)]
pub fn step_batch_1b(
    tbl: &TableW6,
    cur_q: &mut ExtPoint,
    cur_raw: &mut [u8; 32],
    pubs: &mut [[u8; 32]; STEP_BATCH],
    privs: &mut [[u8; 32]; STEP_BATCH],
) {
    let mut nums = [ZERO; STEP_BATCH];
    let mut dens = [ZERO; STEP_BATCH];

    for i in 0..STEP_BATCH {
        // Q ← Q + B  (one Niels mixed addition: 7 field muls)
        *cur_q = cur_q.add_niels(&tbl.step_1b);

        // Increment raw 256-bit little-endian counter by 1
        let mut carry = 1u64;
        for b in cur_raw.iter_mut() {
            let sum = *b as u64 + carry;
            *b = sum as u8;
            carry = sum >> 8;
            if carry == 0 {
                break;
            }
        }

        // Clamp the raw counter to produce a valid X25519 private key
        let mut clamped = *cur_raw;
        clamped[0]  &= 248;   // clear low 3 bits
        clamped[31] &= 127;   // clear bit 255
        clamped[31] |= 64;    // set  bit 254
        privs[i] = clamped;

        // Collect numerator / denominator for the birational map
        //   u = (Z + Y) / (Z - Y)
        nums[i] = cur_q.z.add(cur_q.y);
        dens[i] = cur_q.z.sub(cur_q.y);
    }

    // Montgomery Batch Inversion on STEP_BATCH elements
    let mut prefix_prods = [ZERO; STEP_BATCH];
    prefix_prods[0] = dens[0];
    for i in 1..STEP_BATCH {
        prefix_prods[i] = prefix_prods[i - 1].mul(dens[i]);
    }

    let mut inv_all = prefix_prods[STEP_BATCH - 1].invert();

    let mut den_invs = [ZERO; STEP_BATCH];
    for i in (1..STEP_BATCH).rev() {
        den_invs[i] = inv_all.mul(prefix_prods[i - 1]);
        inv_all = inv_all.mul(dens[i]);
    }
    den_invs[0] = inv_all;

    for i in 0..STEP_BATCH {
        pubs[i] = nums[i].mul(den_invs[i]).encode();
    }
}


// ══════════════════════════════════════════════════════════════════════════════
// 4-Way Interleaved Lane Stepping Engine (THE HOT PATH)
// ══════════════════════════════════════════════════════════════════════════════

/// Number of independent Q chains run simultaneously per thread.
/// Zero data-dependencies across lanes → CPU pipelines all 4×7=28 field muls simultaneously.
pub const LANES: usize = 4;

/// Total outputs per `step_batch_4lanes` call.
pub const LANE_BATCH: usize = LANES * STEP_BATCH;

/// Pre-allocated scratch space for `step_batch_4lanes`.
/// Allocate once per worker thread, reuse every batch call — zero hot-path allocs.
pub struct LaneScratch {
    pub nums:         Box<[Fe; LANE_BATCH]>,
    pub dens:         Box<[Fe; LANE_BATCH]>,
    pub prefix_prods: Box<[Fe; LANE_BATCH]>,
    pub den_invs:     Box<[Fe; LANE_BATCH]>,
}

impl LaneScratch {
    pub fn new() -> Box<Self> {
        Box::new(Self {
            nums:         vec![ZERO; LANE_BATCH].into_boxed_slice().try_into().unwrap(),
            dens:         vec![ZERO; LANE_BATCH].into_boxed_slice().try_into().unwrap(),
            prefix_prods: vec![ZERO; LANE_BATCH].into_boxed_slice().try_into().unwrap(),
            den_invs:     vec![ZERO; LANE_BATCH].into_boxed_slice().try_into().unwrap(),
        })
    }
}

/// 4-way interleaved incremental stepping — the primary hot path.
///
/// # Arguments
/// * `tbl`      — precomputed table (only `step_8b` is used here)
/// * `qs`       — [in/out] 4 independent EC point states
/// * `cur_privs`— [in/out] 4 independent 256-bit little-endian clamped counters
/// * `scratch`  — pre-allocated intermediate buffers (zero alloc in hot path)
/// * `pubs`     — [out] `LANE_BATCH` public keys, lane 0 first then lane 1…
/// * `privs`    — [out] `LANE_BATCH` private keys, same ordering
#[inline(never)]
pub fn step_batch_4lanes(
    tbl: &TableW6,
    qs: &mut [ExtPoint; LANES],
    cur_privs: &mut [[u8; 32]; LANES],
    scratch: &mut LaneScratch,
    pubs: &mut [[u8; 32]; LANE_BATCH],
    privs: &mut [[u8; 32]; LANE_BATCH],
) {
    // ── Inner loop: advance all 4 chains simultaneously ──────────────────────
    // The 4 add_niels calls are completely independent — the CPU issues them
    // in parallel using its OOO execution units.
    for i in 0..STEP_BATCH {
        // Advance all 4 chains (28 independent field muls, fully pipelined)
        qs[0] = qs[0].add_niels(&tbl.step_8b);
        qs[1] = qs[1].add_niels(&tbl.step_8b);
        qs[2] = qs[2].add_niels(&tbl.step_8b);
        qs[3] = qs[3].add_niels(&tbl.step_8b);

        // Advance all 4 private key counters by 8
        for lane in 0..LANES {
            let mut carry = 8u64;
            for b in cur_privs[lane].iter_mut() {
                let sum = *b as u64 + carry;
                *b = sum as u8;
                carry = sum >> 8;
                if carry == 0 { break; }
            }
            let base = lane * STEP_BATCH + i;
            privs[base]              = cur_privs[lane];
            scratch.nums[base] = qs[lane].z.add(qs[lane].y);
            scratch.dens[base] = qs[lane].z.sub(qs[lane].y);
        }
    }

    // ── Single Montgomery Batch Inversion over all LANE_BATCH denominators ───
    scratch.prefix_prods[0] = scratch.dens[0];
    for i in 1..LANE_BATCH {
        scratch.prefix_prods[i] = scratch.prefix_prods[i - 1].mul(scratch.dens[i]);
    }

    let mut inv_all = scratch.prefix_prods[LANE_BATCH - 1].invert();

    for i in (1..LANE_BATCH).rev() {
        scratch.den_invs[i] = inv_all.mul(scratch.prefix_prods[i - 1]);
        inv_all = inv_all.mul(scratch.dens[i]);
    }
    scratch.den_invs[0] = inv_all;

    for i in 0..LANE_BATCH {
        pubs[i] = scratch.nums[i].mul(scratch.den_invs[i]).encode();
    }
}

/// 4-way interleaved stepping optimized for vanity search.
/// Advances 4 lanes, batch-inverts, applies byte-0 filter, and calls `on_candidate`
/// for any key whose prefix score meets the threshold.
/// Private keys are written to `privs[i]` for all i so the closure can use them.
///
/// The callback signature is `(batch_idx, prefix_idx, score, pub_bytes, priv_bytes)`.
#[inline(never)]
pub fn step_batch_4lanes_vanity(
    tbl: &TableW6,
    qs: &mut [ExtPoint; LANES],
    cur_privs: &mut [[u8; 32]; LANES],
    scratch: &mut LaneScratch,
    privs: &mut [[u8; 32]; LANE_BATCH],
    byte0_filter: &[bool; 256],
    skip_byte0_check: bool,
    specs: &[PrefixSpec],
    thresh: usize,
    on_candidate: &mut impl FnMut(usize, usize, usize, &[u8; 32], &[u8; 32]), // (batch_idx, pfx_idx, score, pub, priv)
) {
    // ── Inner loop: advance all 4 chains simultaneously ──────────────────────
    for i in 0..STEP_BATCH {
        qs[0] = qs[0].add_niels(&tbl.step_8b);
        qs[1] = qs[1].add_niels(&tbl.step_8b);
        qs[2] = qs[2].add_niels(&tbl.step_8b);
        qs[3] = qs[3].add_niels(&tbl.step_8b);

        for lane in 0..LANES {
            let mut carry = 8u64;
            for b in cur_privs[lane].iter_mut() {
                let sum = *b as u64 + carry;
                *b = sum as u8;
                carry = sum >> 8;
                if carry == 0 { break; }
            }
            let base = lane * STEP_BATCH + i;
            privs[base] = cur_privs[lane];
            scratch.nums[base] = qs[lane].z.add(qs[lane].y);
            scratch.dens[base] = qs[lane].z.sub(qs[lane].y);
        }
    }

    // ── Single Montgomery Batch Inversion over all LANE_BATCH denominators ───
    scratch.prefix_prods[0] = scratch.dens[0];
    for i in 1..LANE_BATCH {
        scratch.prefix_prods[i] = scratch.prefix_prods[i - 1].mul(scratch.dens[i]);
    }

    let mut inv_all = scratch.prefix_prods[LANE_BATCH - 1].invert();

    for i in (1..LANE_BATCH).rev() {
        scratch.den_invs[i] = inv_all.mul(scratch.prefix_prods[i - 1]);
        inv_all = inv_all.mul(scratch.dens[i]);
    }
    scratch.den_invs[0] = inv_all;

    // ── Early Byte-0 reject & on-the-fly multi-prefix match ──────────────────
    for i in 0..LANE_BATCH {
        let u_fe = scratch.nums[i].mul(scratch.den_invs[i]);
        if !skip_byte0_check {
            let b0 = u_fe.encode_byte0();
            if !byte0_filter[b0 as usize] {
                continue;
            }
        }

        // Passed byte-0 filter → full encode and check all prefixes
        let pub_key = u_fe.encode();
        // Find the prefix with the best score
        let mut best_score = 0usize;
        let mut best_pfx = 0usize;
        for (pi, spec) in specs.iter().enumerate() {
            let m = spec.match_score(&pub_key);
            if m > best_score {
                best_score = m;
                best_pfx = pi;
            }
        }
        let pfx_len = specs[best_pfx].text.len();
        if best_score >= thresh || best_score == pfx_len {
            on_candidate(i, best_pfx, best_score, &pub_key, &privs[i]);
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Vanity & Long-Zero Search Logic
// ══════════════════════════════════════════════════════════════════════════════

const BASE64_CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
const TOP_N: usize = 5;

static THRESHOLD: AtomicI32 = AtomicI32::new(0);
static TOTAL: AtomicU64 = AtomicU64::new(0);
static TOP_LOCK: OnceLock<Mutex<Vec<Entry>>> = OnceLock::new();

fn get_top() -> &'static Mutex<Vec<Entry>> {
    TOP_LOCK.get_or_init(|| Mutex::new(Vec::new()))
}

fn zero_run(b: &[u8]) -> usize {
    let mut best = 0;
    let mut cur = 0;
    for &x in b {
        if x == 0 {
            cur += 1;
            if cur > best {
                best = cur;
            }
        } else {
            cur = 0;
        }
    }
    best
}

#[inline(always)]
fn match_char(a: u8, b: u8) -> bool {
    let lc = |c: u8| {
        if c.is_ascii_uppercase() {
            c + 32
        } else {
            c
        }
    };
    let (a, b) = (lc(a), lc(b));
    if a == b {
        return true;
    }
    matches!(
        (a, b),
        (b'e', b'3')
            | (b'3', b'e')
            | (b'a', b'4')
            | (b'4', b'a')
            | (b's', b'5')
            | (b'5', b's')
            | (b'g', b'9')
            | (b'9', b'g')
            | (b'i', b'1')
            | (b'1', b'i')
            | (b'o', b'0')
            | (b'0', b'o')
            | (b'b', b'8')
            | (b'8', b'b')
            | (b'z', b'2')
            | (b'2', b'z')
    )
}

#[inline(always)]
fn match_prefix_len(s: &[u8], pfx: &[u8]) -> usize {
    (0..s.len().min(pfx.len()))
        .take_while(|&i| match_char(s[i], pfx[i]))
        .count()
}

/// Matches the public key against a Base64 prefix *directly from the raw 32 bytes*
/// without calling `base64::encode` or writing to a temporary ASCII buffer.
#[inline(always)]
fn match_pub_prefix(pub_bytes: &[u8; 32], pfx: &[u8]) -> usize {
    let pfx_len = pfx.len();
    let mut matched = 0;
    let mut i = 0; // byte index in pub_bytes
    while i + 3 <= 32 && matched < pfx_len {
        let b0 = pub_bytes[i] as u32;
        let b1 = pub_bytes[i + 1] as u32;
        let b2 = pub_bytes[i + 2] as u32;
        let triple = (b0 << 16) | (b1 << 8) | b2;

        let c0 = BASE64_CHARS[((triple >> 18) & 0x3F) as usize];
        if !match_char(c0, pfx[matched]) { return matched; }
        matched += 1;
        if matched == pfx_len { return matched; }

        let c1 = BASE64_CHARS[((triple >> 12) & 0x3F) as usize];
        if !match_char(c1, pfx[matched]) { return matched; }
        matched += 1;
        if matched == pfx_len { return matched; }

        let c2 = BASE64_CHARS[((triple >> 6) & 0x3F) as usize];
        if !match_char(c2, pfx[matched]) { return matched; }
        matched += 1;
        if matched == pfx_len { return matched; }

        let c3 = BASE64_CHARS[(triple & 0x3F) as usize];
        if !match_char(c3, pfx[matched]) { return matched; }
        matched += 1;
        if matched == pfx_len { return matched; }

        i += 3;
    }

    // Remainder: bytes 30 and 31 (2 bytes -> 3 Base64 chars + 1 pad char)
    if i < 32 && matched < pfx_len {
        let b0 = pub_bytes[30] as u32;
        let b1 = pub_bytes[31] as u32;
        let pair = (b0 << 16) | (b1 << 8);

        let c0 = BASE64_CHARS[((pair >> 18) & 0x3F) as usize];
        if !match_char(c0, pfx[matched]) { return matched; }
        matched += 1;
        if matched == pfx_len { return matched; }

        let c1 = BASE64_CHARS[((pair >> 12) & 0x3F) as usize];
        if !match_char(c1, pfx[matched]) { return matched; }
        matched += 1;
        if matched == pfx_len { return matched; }

        let c2 = BASE64_CHARS[((pair >> 6) & 0x3F) as usize];
        if !match_char(c2, pfx[matched]) { return matched; }
        matched += 1;
    }

    matched
}

/// Case-sensitive variant: the Base64 character must equal the prefix byte exactly (no leet/case folding).
#[inline(always)]
fn match_pub_prefix_cs(pub_bytes: &[u8; 32], pfx: &[u8]) -> usize {
    let pfx_len = pfx.len();
    let mut matched = 0;
    let mut i = 0;
    while i + 3 <= 32 && matched < pfx_len {
        let b0 = pub_bytes[i] as u32;
        let b1 = pub_bytes[i + 1] as u32;
        let b2 = pub_bytes[i + 2] as u32;
        let triple = (b0 << 16) | (b1 << 8) | b2;

        let c0 = BASE64_CHARS[((triple >> 18) & 0x3F) as usize];
        if c0 != pfx[matched] { return matched; }
        matched += 1;
        if matched == pfx_len { return matched; }

        let c1 = BASE64_CHARS[((triple >> 12) & 0x3F) as usize];
        if c1 != pfx[matched] { return matched; }
        matched += 1;
        if matched == pfx_len { return matched; }

        let c2 = BASE64_CHARS[((triple >> 6) & 0x3F) as usize];
        if c2 != pfx[matched] { return matched; }
        matched += 1;
        if matched == pfx_len { return matched; }

        let c3 = BASE64_CHARS[(triple & 0x3F) as usize];
        if c3 != pfx[matched] { return matched; }
        matched += 1;
        if matched == pfx_len { return matched; }

        i += 3;
    }

    if i < 32 && matched < pfx_len {
        let b0 = pub_bytes[30] as u32;
        let b1 = pub_bytes[31] as u32;
        let pair = (b0 << 16) | (b1 << 8);

        let c0 = BASE64_CHARS[((pair >> 18) & 0x3F) as usize];
        if c0 != pfx[matched] { return matched; }
        matched += 1;
        if matched == pfx_len { return matched; }

        let c1 = BASE64_CHARS[((pair >> 12) & 0x3F) as usize];
        if c1 != pfx[matched] { return matched; }
        matched += 1;
        if matched == pfx_len { return matched; }

        let c2 = BASE64_CHARS[((pair >> 6) & 0x3F) as usize];
        if c2 != pfx[matched] { return matched; }
        matched += 1;
    }

    matched
}

fn match_choices(c: u8) -> usize {
    BASE64_CHARS.iter().filter(|&&x| match_char(x, c)).count()
}

// ══════════════════════════════════════════════════════════════════════════════
// Multi-Prefix Specification
// ══════════════════════════════════════════════════════════════════════════════

/// One entry in the prefix search list, e.g. `s'Catflare'` or `i'LilyCC'`.
#[derive(Clone, Debug)]
pub struct PrefixSpec {
    /// Raw bytes to match in Base64 output.
    /// For case-insensitive specs this is stored lowercase.
    pub text: Vec<u8>,
    /// `true`  → case-sensitive exact match (`s'...'`)
    /// `false` → case-insensitive + leetspeak (`i'...'`)
    pub case_sensitive: bool,
}

impl PrefixSpec {
    #[inline(always)]
    pub fn match_score(&self, pub_bytes: &[u8; 32]) -> usize {
        if self.case_sensitive {
            match_pub_prefix_cs(pub_bytes, &self.text)
        } else {
            match_pub_prefix(pub_bytes, &self.text)
        }
    }

    /// Human-readable label, e.g. `s'Catflare'` or `i'lilycc'`.
    pub fn display(&self) -> String {
        let mode = if self.case_sensitive { "s" } else { "i" };
        format!("{}'{}' ", mode, String::from_utf8_lossy(&self.text))
    }
}

/// Parse a prefix list in the format `[s'Foo', i'Bar']` or a plain bare word
/// (legacy single-prefix, treated as case-insensitive + leetspeak).
///
/// Accepted item formats inside brackets:
///   `s'...'`   — case-sensitive exact match
///   `i'...'`   — case-insensitive + leetspeak
///   `'...'`    — case-insensitive (shorthand, no mode letter)
///   bare word  — case-insensitive (shorthand)
pub fn parse_prefix_list(input: &str) -> Result<Vec<PrefixSpec>, String> {
    let s = input.trim();
    if s.starts_with('[') {
        // Bracketed list
        let inner = s
            .strip_prefix('[')
            .and_then(|s| s.strip_suffix(']'))
            .ok_or_else(|| "prefix list must be enclosed in [ ]".to_string())?
            .trim();
        if inner.is_empty() {
            return Err("prefix list is empty".to_string());
        }
        // Split on commas that are NOT inside single-quotes
        let mut items: Vec<&str> = Vec::new();
        let mut start = 0;
        let mut in_quote = false;
        for (i, ch) in inner.char_indices() {
            match ch {
                '\'' => in_quote = !in_quote,
                ',' if !in_quote => {
                    items.push(&inner[start..i]);
                    start = i + 1;
                }
                _ => {}
            }
        }
        items.push(&inner[start..]);

        let mut specs = Vec::new();
        for item in items {
            let item = item.trim();
            if item.is_empty() { continue; }
            specs.push(parse_one_item(item)?);
        }
        if specs.is_empty() {
            return Err("prefix list contained no valid items".to_string());
        }
        Ok(specs)
    } else {
        // Legacy bare word — single case-insensitive prefix
        Ok(vec![PrefixSpec {
            text: s.to_lowercase().into_bytes(),
            case_sensitive: false,
        }])
    }
}

fn parse_one_item(item: &str) -> Result<PrefixSpec, String> {
    if item.len() >= 2 && (item.starts_with("s'") || item.starts_with("S'")) {
        let rest = &item[2..];
        let text_str = rest
            .strip_suffix('\'')
            .ok_or_else(|| format!("unclosed quote in prefix item: {:?}", item))?;
        Ok(PrefixSpec { text: text_str.as_bytes().to_vec(), case_sensitive: true })
    } else if item.len() >= 2 && (item.starts_with("i'") || item.starts_with("I'")) {
        let rest = &item[2..];
        let text_str = rest
            .strip_suffix('\'')
            .ok_or_else(|| format!("unclosed quote in prefix item: {:?}", item))?;
        Ok(PrefixSpec { text: text_str.to_lowercase().into_bytes(), case_sensitive: false })
    } else if item.starts_with('\'') && item.ends_with('\'') && item.len() >= 2 {
        let text_str = &item[1..item.len() - 1];
        Ok(PrefixSpec { text: text_str.to_lowercase().into_bytes(), case_sensitive: false })
    } else {
        // bare word
        Ok(PrefixSpec { text: item.to_lowercase().into_bytes(), case_sensitive: false })
    }
}

/// Byte-0 filter covering all prefixes — passes if the raw byte could be the
/// start of ANY prefix in the list.
pub fn build_byte0_filter_multi(specs: &[PrefixSpec]) -> [bool; 256] {
    let mut filter = [false; 256];
    for spec in specs {
        if spec.text.is_empty() { continue; }
        let target_char = spec.text[0];
        for (b0, item) in filter.iter_mut().enumerate() {
            let b64_char0 = BASE64_CHARS[b0 >> 2];
            let matches = if spec.case_sensitive {
                b64_char0 == target_char
            } else {
                match_char(b64_char0, target_char)
            };
            if matches { *item = true; }
        }
    }
    filter
}

#[derive(Clone)]
struct Entry {
    pub_b64: String,
    priv_b64: String,
    score: usize,
    pfx_idx: usize, // index into the PrefixSpec list that this entry matched
}

fn record(e: Entry) {
    let mut top = get_top().lock().unwrap();
    if top.len() == TOP_N && e.score <= top[TOP_N - 1].score {
        return;
    }
    let i = top.partition_point(|x| x.score > e.score);
    top.insert(i, e);
    if top.len() > TOP_N {
        top.truncate(TOP_N);
    }
    if top.len() == TOP_N {
        THRESHOLD.store(top[TOP_N - 1].score as i32 + 1, Ordering::Relaxed);
    }
}

fn render_vanity(specs: &[PrefixSpec], expected: f64, start: Instant) {
    let snap = get_top().lock().unwrap().clone();
    let t = TOTAL.load(Ordering::Relaxed);
    let el = start.elapsed().as_secs_f64();
    print!("\x1b[H\x1b[J");
    // Show all targets
    print!("Targets: ");
    for spec in specs {
        let mode = if spec.case_sensitive { "case-sensitive" } else { "case-insensitive+leet" };
        print!("{}({}) ", String::from_utf8_lossy(&spec.text), mode);
    }
    println!();
    println!(
        "Tries: {}  |  {:.0} keys/s  |  {:.0}s elapsed",
        t,
        t as f64 / el.max(0.001),
        el
    );
    println!(
        "Expected tries for a hit: ~{:.2e} (you're at {:.1}% of that)\n",
        expected,
        t as f64 / expected * 100.0
    );
    println!("Closest so far:");
    for e in &snap {
        let spec = &specs[e.pfx_idx.min(specs.len() - 1)];
        let pfx_len = spec.text.len();
        let mode_tag = if spec.case_sensitive { "s" } else { "i" };
        let m = e.score.min(e.pub_b64.len());
        println!(
            "  [{}/{}] {}'{}'  \x1b[32m{}\x1b[0m{}",
            e.score,
            pfx_len,
            mode_tag,
            String::from_utf8_lossy(&spec.text),
            &e.pub_b64[..m],
            &e.pub_b64[m..]
        );
        println!("         priv: {}", e.priv_b64);
    }
}

fn render_long(start: Instant) {
    let snap = get_top().lock().unwrap().clone();
    let t = TOTAL.load(Ordering::Relaxed);
    let el = start.elapsed().as_secs_f64();
    print!("\x1b[H\x1b[J");
    println!("Mode: longest consecutive zero-byte run in private key");
    println!(
        "Tries: {}  |  {:.0} keys/s  |  {:.0}s elapsed\n",
        t,
        t as f64 / el.max(0.001),
        el
    );
    println!("Best keys so far:");
    for (i, e) in snap.iter().enumerate() {
        println!("  #{}  [{} consecutive zero bytes]", i + 1, e.score);
        println!("      priv: {}", e.priv_b64);
        println!("      pub:  {}", e.pub_b64);
    }
}

fn worker_vanity_stepped(
    tbl: &TableW6,
    specs: Vec<PrefixSpec>,
    byte0_filter: [bool; 256],
    tx: std::sync::mpsc::Sender<Entry>,
) {
    let mut rng = OsRng;
    let mut privs_box: Box<[[u8; 32]; LANE_BATCH]> = vec![[0u8; 32]; LANE_BATCH].into_boxed_slice().try_into().unwrap();
    let mut scratch = LaneScratch::new();
    let mut local: u64 = 0;
    let mut found = false;

    'outer: loop {
        let mut cur_privs = [[0u8; 32]; LANES];
        for p in cur_privs.iter_mut() {
            rng.fill_bytes(p);
            p[0]  &= 248;
            p[31] &= 127;
            p[31] |= 64;
        }

        let mut qs: [ExtPoint; LANES] = [
            scalar_mult_w6(tbl, &cur_privs[0]),
            scalar_mult_w6(tbl, &cur_privs[1]),
            scalar_mult_w6(tbl, &cur_privs[2]),
            scalar_mult_w6(tbl, &cur_privs[3]),
        ];

        for _ in 0..64 {
            let thresh = THRESHOLD.load(Ordering::Relaxed) as usize;
            let skip_byte0 = thresh <= 1;

            step_batch_4lanes_vanity(
                tbl,
                &mut qs,
                &mut cur_privs,
                &mut scratch,
                &mut *privs_box,
                &byte0_filter,
                skip_byte0,
                &specs,
                thresh,
                &mut |_i, pfx_idx, m, pub_key, priv_key| {
                    let pfx_len = specs[pfx_idx].text.len();
                    let e = Entry {
                        pub_b64:  B64.encode(pub_key),
                        priv_b64: B64.encode(priv_key),
                        score: m,
                        pfx_idx,
                    };
                    record(e.clone());
                    if m == pfx_len {
                        let _ = tx.send(e);
                        found = true;
                    }
                },
            );

            if found { break 'outer; }

            local += LANE_BATCH as u64;
            if local >= 65536 {
                TOTAL.fetch_add(local, Ordering::Relaxed);
                local = 0;
            }
        }
    }
}


fn worker_long_stepped(tbl: &TableW6) {
    let mut rng = OsRng;
    let mut pubs_box:  Box<[[u8; 32]; LANE_BATCH]> = vec![[0u8; 32]; LANE_BATCH].into_boxed_slice().try_into().unwrap();
    let mut privs_box: Box<[[u8; 32]; LANE_BATCH]> = vec![[0u8; 32]; LANE_BATCH].into_boxed_slice().try_into().unwrap();
    let mut scratch = LaneScratch::new();
    let mut local: u64 = 0;

    loop {
        let mut cur_privs = [[0u8; 32]; LANES];
        for p in cur_privs.iter_mut() {
            rng.fill_bytes(p);
            p[0]  &= 248;
            p[31] &= 127;
            p[31] |= 64;
        }

        let mut qs: [ExtPoint; LANES] = [
            scalar_mult_w6(tbl, &cur_privs[0]),
            scalar_mult_w6(tbl, &cur_privs[1]),
            scalar_mult_w6(tbl, &cur_privs[2]),
            scalar_mult_w6(tbl, &cur_privs[3]),
        ];

        for _ in 0..64 {
            step_batch_4lanes(tbl, &mut qs, &mut cur_privs, &mut scratch, &mut *pubs_box, &mut *privs_box);
            let pubs  = &*pubs_box;
            let privs = &*privs_box;


            let thresh = THRESHOLD.load(Ordering::Relaxed) as usize;

            for i in 0..LANE_BATCH {
                let m = zero_run(&privs[i]);
                if m >= thresh {
                    record(Entry {
                        pub_b64:  B64.encode(pubs[i]),
                        priv_b64: B64.encode(privs[i]),
                        score: m,
                        pfx_idx: 0,
                    });
                }
            }

            local += LANE_BATCH as u64;
            if local >= 65536 {
                TOTAL.fetch_add(local, Ordering::Relaxed);
                local = 0;
            }
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Benchmark Mode
// ══════════════════════════════════════════════════════════════════════════════

fn run_benchmark(tbl: &TableW6, n: usize) {
    let mut rng = OsRng;
    let mut key = [0u8; 32];

    // Single-key mode benchmark (from-scratch scalar mult)
    for _ in 0..32 {
        rng.fill_bytes(&mut key);
        std::hint::black_box(x25519_single(tbl, &key));
    }

    let mut times: Vec<f64> = Vec::with_capacity(n);
    for _ in 0..n {
        rng.fill_bytes(&mut key);
        let t0 = Instant::now();
        std::hint::black_box(x25519_single(tbl, &key));
        times.push(t0.elapsed().as_nanos() as f64);
    }

    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let sum: f64 = times.iter().sum();
    let n_f = n as f64;
    let avg = sum / n_f;
    let min = times[0];
    let max = times[n - 1];
    let median = times[n / 2];

    println!("=== Rust X25519 From-Scratch Single-Key (w=6, {} keys) ===", n);
    println!("  Min:    {:10.2} ns  ({:.3} µs)", min, min / 1000.0);
    println!("  Max:    {:10.2} ns  ({:.3} µs)", max, max / 1000.0);
    println!("  Avg:    {:10.2} ns  ({:.3} µs)", avg, avg / 1000.0);
    println!("  Median: {:10.2} ns  ({:.3} µs)", median, median / 1000.0);
    println!(
        "  Throughput: {:.0} keys/s ({:.3} Mkeys/s)\n",
        1e9 / avg,
        1e9 / avg / 1e6
    );

    // ── Single-chain +8B stepping benchmark ─────────────────────────────────
    let batches = (n.max(STEP_BATCH) * 40) / STEP_BATCH;
    let mut b_pubs  = vec![[0u8; 32]; LANE_BATCH]; // sized for largest benchmark
    let mut b_privs = vec![[0u8; 32]; LANE_BATCH];
    let b_pubs_s  = b_pubs.as_mut_slice()[..STEP_BATCH].as_mut_ptr();
    let b_privs_s = b_privs.as_mut_slice()[..STEP_BATCH].as_mut_ptr();
    // SAFETY: slices are exactly STEP_BATCH elements and outlive the borrow
    let b_pubs_arr:  &mut [[u8;32]; STEP_BATCH] = unsafe { &mut *(b_pubs_s  as *mut _) };
    let b_privs_arr: &mut [[u8;32]; STEP_BATCH] = unsafe { &mut *(b_privs_s as *mut _) };

    let mut seed_key = [0u8; 32];
    rng.fill_bytes(&mut seed_key);
    seed_key[0]  &= 248;
    seed_key[31] &= 127;
    seed_key[31] |= 64;

    let mut cur_q    = scalar_mult_w6(tbl, &seed_key);
    let mut cur_priv = seed_key;

    for _ in 0..5 {
        step_batch_256(tbl, &mut cur_q, &mut cur_priv, b_pubs_arr, b_privs_arr);
    }
    let t0 = Instant::now();
    for _ in 0..batches {
        step_batch_256(tbl, &mut cur_q, &mut cur_priv, b_pubs_arr, b_privs_arr);
    }
    let total_8b   = batches * STEP_BATCH;
    let elapsed_8b = t0.elapsed();
    let ns_8b      = elapsed_8b.as_nanos() as f64 / total_8b as f64;
    let tp_8b      = total_8b as f64 / elapsed_8b.as_secs_f64();

    println!("=== Incremental Stepping +8B (single chain, {} keys) ===", total_8b);
    println!("  Cost per key: {:.2} ns  ({:.3} µs)", ns_8b, ns_8b / 1000.0);
    println!("  Single-Core Throughput: {:.0} keys/s ({:.3} Mkeys/s)", tp_8b, tp_8b / 1e6);
    println!("  Projected 12-Thread:    ~{:.1} Mkeys/s\n", (tp_8b * 12.0) / 1e6);

    // ── 4-lane interleaved +8B stepping benchmark (THE HOT PATH) ────────────
    let lane_batches = (n.max(LANE_BATCH) * 10) / LANE_BATCH;

    let b_pubs_l:  &mut [[u8;32]; LANE_BATCH] = unsafe { &mut *(b_pubs.as_mut_ptr()  as *mut _) };
    let b_privs_l: &mut [[u8;32]; LANE_BATCH] = unsafe { &mut *(b_privs.as_mut_ptr() as *mut _) };

    let mut cp4 = [[0u8; 32]; LANES];
    for p in cp4.iter_mut() {
        rng.fill_bytes(p);
        p[0]  &= 248;
        p[31] &= 127;
        p[31] |= 64;
    }

    let mut qs4: [ExtPoint; LANES] = [
        scalar_mult_w6(tbl, &cp4[0]),
        scalar_mult_w6(tbl, &cp4[1]),
        scalar_mult_w6(tbl, &cp4[2]),
        scalar_mult_w6(tbl, &cp4[3]),
    ];

    let mut bench_scratch = LaneScratch::new();

    for _ in 0..5 {
        step_batch_4lanes(tbl, &mut qs4, &mut cp4, &mut bench_scratch, b_pubs_l, b_privs_l);
    }
    let t2 = Instant::now();
    for _ in 0..lane_batches {
        step_batch_4lanes(tbl, &mut qs4, &mut cp4, &mut bench_scratch, b_pubs_l, b_privs_l);
    }
    let total_4l   = lane_batches * LANE_BATCH;
    let elapsed_4l = t2.elapsed();
    let ns_4l      = elapsed_4l.as_nanos() as f64 / total_4l as f64;
    let tp_4l      = total_4l as f64 / elapsed_4l.as_secs_f64();

    println!("=== 4-Lane Interleaved +8B (hot path, {} keys) ===", total_4l);
    println!("  Cost per key: {:.2} ns  ({:.3} µs)", ns_4l, ns_4l / 1000.0);
    println!("  Single-Core Throughput: {:.0} keys/s ({:.3} Mkeys/s)", tp_4l, tp_4l / 1e6);
    println!("  Projected 12-Thread:    ~{:.1} Mkeys/s  ← ACTUAL SEARCH SPEED", (tp_4l * 12.0) / 1e6);
}


// ══════════════════════════════════════════════════════════════════════════════
// Main
// ══════════════════════════════════════════════════════════════════════════════

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let has = |flag: &str| args.iter().any(|a| a == flag);
    let get_usize = |flag: &str, default: usize| -> usize {
        args.iter()
            .position(|a| a == flag)
            .and_then(|p| args.get(p + 1))
            .and_then(|s| s.parse().ok())
            .unwrap_or(default)
    };

    let tbl = build_table_w6();

    if has("--bench") {
        run_benchmark(&tbl, get_usize("--bench-n", 128));
        return;
    }

    if has("--find-long") {
        THRESHOLD.store(0, Ordering::Relaxed);
        let cpus = num_cpus();
        let ptr = &*tbl as *const TableW6 as usize;
        for _ in 0..cpus {
            std::thread::spawn(move || {
                worker_long_stepped(unsafe { &*(ptr as *const TableW6) });
            });
        }
        let start = Instant::now();
        loop {
            std::thread::sleep(Duration::from_millis(250));
            render_long(start);
        }
    }

    let positional: Vec<&String> = args
        .iter()
        .skip(1)
        .filter(|a| !a.starts_with('-'))
        .collect();
    if positional.is_empty() {
        println!("usage:");
        println!("  vanity25519 <prefix>");
        println!("  vanity25519 \"[s'Catflare', i'LilyCC']\"");
        println!("  vanity25519 --find-long");
        println!("  vanity25519 --bench [--bench-n N]");
        println!();
        println!("Prefix item formats (inside [ ]):");
        println!("  s'Text'   case-sensitive exact match");
        println!("  i'Text'   case-insensitive + leetspeak (default)");
        println!("  'Text'    same as i'Text'");
        println!("  Text      same as i'Text' (bare word, legacy)");
        return;
    }

    // Join positional args so users can write the list without quoting the whole thing
    let raw_input = positional.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(" ");
    let specs = match parse_prefix_list(&raw_input) {
        Ok(s) => s,
        Err(e) => {
            println!("Error parsing prefix list: {}", e);
            return;
        }
    };

    // Validate all specs and compute expected difficulty (use easiest prefix = smallest expected)
    let mut expected = f64::INFINITY;
    for spec in &specs {
        if spec.text.len() > 43 {
            println!("prefix '{}' is too long (max 43 chars)", String::from_utf8_lossy(&spec.text));
            return;
        }
        if spec.text.is_empty() {
            println!("prefix cannot be empty");
            return;
        }
        let mut e = 1.0f64;
        for &c in &spec.text {
            let ch = if spec.case_sensitive {
                // Exact match: only 1 Base64 char per position
                if BASE64_CHARS.contains(&c) { 1 } else { 0 }
            } else {
                match_choices(c)
            };
            if ch == 0 {
                println!("invalid vanity character {:?} in prefix '{}'", c as char, String::from_utf8_lossy(&spec.text));
                return;
            }
            e *= 64.0 / ch as f64;
        }
        if e < expected { expected = e; }
    }

    let byte0_filter = build_byte0_filter_multi(&specs);

    THRESHOLD.store(1, Ordering::Relaxed);
    let (tx, rx) = std::sync::mpsc::channel::<Entry>();
    let ptr = &*tbl as *const TableW6 as usize;
    for _ in 0..num_cpus() {
        let (tx2, specs2) = (tx.clone(), specs.clone());
        std::thread::spawn(move || {
            worker_vanity_stepped(
                unsafe { &*(ptr as *const TableW6) },
                specs2,
                byte0_filter,
                tx2,
            );
        });
    }

    let start = Instant::now();
    loop {
        match rx.recv_timeout(Duration::from_millis(250)) {
            Ok(r) => {
                render_vanity(&specs, expected, start);
                let matched_spec = &specs[r.pfx_idx.min(specs.len() - 1)];
                println!("\nFOUND! (matched {})", matched_spec.display().trim());
                println!("private: {}", r.priv_b64);
                println!("public:  {}", r.pub_b64);
                return;
            }
            Err(_) => render_vanity(&specs, expected, start),
        }
    }
}

fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

// ══════════════════════════════════════════════════════════════════════════════
// Tests
// ══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn from_hex(s: &str) -> [u8; 32] {
        let mut out = [0u8; 32];
        for i in 0..32 {
            out[i] = u8::from_str_radix(&s[i * 2..i * 2 + 2], 16).unwrap();
        }
        out
    }
    fn to_hex(b: &[u8; 32]) -> String {
        b.iter().map(|x| format!("{:02x}", x)).collect()
    }

    #[test]
    fn rfc7748_alice() {
        let tbl = build_table_w6();
        let priv_key =
            from_hex("77076d0a7318a57d3c16c17251b26645df4c2f87ebc0992ab177fba51db92c2a");
        let pub_key = x25519_single(&tbl, &priv_key);
        assert_eq!(
            to_hex(&pub_key),
            "8520f0098930a754748b7ddcb43ef75a0dbf3a0d26381af4eba4a98eaa9b4e6a",
            "RFC7748 Alice test vector mismatch"
        );
    }

    #[test]
    fn rfc7748_bob() {
        let tbl = build_table_w6();
        let priv_key =
            from_hex("5dab087e624a8a4b79e17f8b83800ee66f3bb1292618b6fd1c2f8b27ff88e0eb");
        let pub_key = x25519_single(&tbl, &priv_key);
        assert_eq!(
            to_hex(&pub_key),
            "de9edb7d7b7dc1b4d35b61c2ece435373f8343c85b78674dadfc7e146f882b4f",
            "RFC7748 Bob test vector mismatch"
        );
    }

    #[test]
    fn incremental_step_matches_full_mult() {
        let tbl = build_table_w6();
        let mut seed_key = [0u8; 32];
        let mut rng = OsRng;
        rng.fill_bytes(&mut seed_key);
        seed_key[0] &= 248;
        seed_key[31] &= 127;
        seed_key[31] |= 64;

        let mut cur_q = scalar_mult_w6(&tbl, &seed_key);
        let mut cur_priv = seed_key;
        let mut pubs: Box<[[u8; 32]; STEP_BATCH]> = vec![[0u8; 32]; STEP_BATCH].into_boxed_slice().try_into().unwrap();
        let mut privs: Box<[[u8; 32]; STEP_BATCH]> = vec![[0u8; 32]; STEP_BATCH].into_boxed_slice().try_into().unwrap();

        step_batch_256(&tbl, &mut cur_q, &mut cur_priv, &mut *pubs, &mut *privs);

        // Verify that sample stepped keys produce the EXACT same public key
        // as independent from-scratch full scalar multiplication!
        for &idx in &[0, 1, 15, 63, 127, 255] {
            let direct_pub = x25519_single(&tbl, &privs[idx]);
            assert_eq!(pubs[idx], direct_pub, "Stepped key mismatch at index {}", idx);
        }
    }
    #[test]
    fn incremental_step_1b_matches_full_mult() {
        let tbl = build_table_w6();
        let mut rng = OsRng;
        let mut seed_key = [0u8; 32];
        rng.fill_bytes(&mut seed_key);
        // Start from a clamped key so Q is well-defined from scalar_mult_w6.
        seed_key[0]  &= 248;
        seed_key[31] &= 127;
        seed_key[31] |= 64;

        // Q = seed_key * B  (seed already clamped, so scalar_mult_w6 is correct)
        let mut cur_q   = scalar_mult_w6(&tbl, &seed_key);
        let mut cur_raw = seed_key; // raw counter = seed (will be incremented by 1 each step)
        let mut pubs: Box<[[u8; 32]; STEP_BATCH]> = vec![[0u8; 32]; STEP_BATCH].into_boxed_slice().try_into().unwrap();
        let mut privs: Box<[[u8; 32]; STEP_BATCH]> = vec![[0u8; 32]; STEP_BATCH].into_boxed_slice().try_into().unwrap();

        // Capture the raw counter values before the step so we can recompute expected.
        // After step_batch_1b, privs[i] = clamp(seed + i+1) and
        // pubs[i] = birational((seed + i+1) * B)  — note: RAW scalar, not clamped.
        step_batch_1b(&tbl, &mut cur_q, &mut cur_raw, &mut *pubs, &mut *privs);

        // Reference: compute expected public key for each raw scalar (seed + i+1)
        // independently via full scalar_mult_w6 + birational map (no clamping).
        let mut raw_counter = seed_key;
        for idx in 0..STEP_BATCH {
            // Increment raw_counter by 1 (same as step_batch_1b does internally)
            let mut carry = 1u64;
            for b in raw_counter.iter_mut() {
                let sum = *b as u64 + carry;
                *b = sum as u8;
                carry = sum >> 8;
                if carry == 0 { break; }
            }

            // Compute expected: (raw_counter * B) → Montgomery u via birational map
            let q_ref = scalar_mult_w6(&tbl, &raw_counter);
            let num   = q_ref.z.add(q_ref.y);
            let den   = q_ref.z.sub(q_ref.y);
            let expected_pub = num.mul(den.invert()).encode();

            if [0, 1, 7, 8, 15, 63, 127, 255].contains(&idx) {
                assert_eq!(
                    pubs[idx], expected_pub,
                    "step_batch_1b pub mismatch at index {}",
                    idx
                );
            }
        }
    }

    #[test]
    fn step_4lanes_matches_full_mult() {
        let tbl = build_table_w6();
        let mut rng = OsRng;

        // Build 4 independent clamped seeds
        let mut seeds = [[0u8; 32]; LANES];
        for s in seeds.iter_mut() {
            rng.fill_bytes(s);
            s[0]  &= 248;
            s[31] &= 127;
            s[31] |= 64;
        }

        let mut qs: [ExtPoint; LANES] = [
            scalar_mult_w6(&tbl, &seeds[0]),
            scalar_mult_w6(&tbl, &seeds[1]),
            scalar_mult_w6(&tbl, &seeds[2]),
            scalar_mult_w6(&tbl, &seeds[3]),
        ];
        let mut cur_privs: [[u8; 32]; LANES] = seeds;

        let mut pubs_box:  Box<[[u8; 32]; LANE_BATCH]> = vec![[0u8; 32]; LANE_BATCH].into_boxed_slice().try_into().unwrap();
        let mut privs_box: Box<[[u8; 32]; LANE_BATCH]> = vec![[0u8; 32]; LANE_BATCH].into_boxed_slice().try_into().unwrap();
        let mut scratch = LaneScratch::new();
        step_batch_4lanes(&tbl, &mut qs, &mut cur_privs, &mut scratch, &mut *pubs_box, &mut *privs_box);
        let pubs  = &*pubs_box;
        let privs = &*privs_box;

        // Each lane's outputs should match x25519_single on the corresponding private key
        for lane in 0..LANES {
            for &rel_idx in &[0usize, 1, 63, 255, STEP_BATCH - 1] {
                let abs_idx = lane * STEP_BATCH + rel_idx;
                let direct  = x25519_single(&tbl, &privs[abs_idx]);
                assert_eq!(
                    pubs[abs_idx], direct,
                    "4-lane pub mismatch: lane={} rel_idx={}",
                    lane, rel_idx
                );
            }
        }
    }

    #[test]
    fn encode_byte0_matches_full_encode() {
        let mut rng = OsRng;
        for _ in 0..1000 {
            let mut b = [0u8; 32];
            rng.fill_bytes(&mut b);
            let fe = Fe::decode(&b);
            let full = fe.encode();
            let b0 = fe.encode_byte0();
            assert_eq!(b0, full[0], "encode_byte0 did not match full encode[0]");
        }
    }

    #[test]
    fn match_pub_prefix_matches_string_prefix() {
        let mut rng = OsRng;
        let test_prefixes = [
            b"cat".to_vec(),
            b"catflare".to_vec(),
            b"42".to_vec(),
            b"wireguard".to_vec(),
            b"zzzz".to_vec(),
        ];
        for _ in 0..500 {
            let mut b = [0u8; 32];
            rng.fill_bytes(&mut b);
            let b64 = B64.encode(b);
            for pfx in &test_prefixes {
                let expected = match_prefix_len(b64.as_bytes(), pfx);
                let actual = match_pub_prefix(&b, pfx);
                assert_eq!(actual, expected, "match_pub_prefix failed for prefix {:?}", pfx);
            }
        }
    }
}
