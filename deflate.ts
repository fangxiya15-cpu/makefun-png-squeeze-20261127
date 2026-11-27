// Optimal-parsing deflate (zopfli-style), written for png-squeeze-01.
// LZ77 with hash chains + dynamic-programming optimal parse (fixed-cost pass, then
// dynamic-Huffman-cost pass, matches cached between passes) + length-limited dynamic
// Huffman (zlib-style overflow fix, max 15, with a Kraft safety net). Deterministic:
// fixed work, no clock, no environment reads.
import { inflateSync } from 'node:zlib';

// ---------- RFC 1951 tables ----------
const LEN_BASE = [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258];
const LEN_EXTRA = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0];
const DIST_BASE = [1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577];
const DIST_EXTRA = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13];
const LEN_SYM = new Uint16Array(259); // NB: symbols run to 285, must not be Uint8
const LEN_EXT = new Uint8Array(259);
for (let s = 0; s < 29; s++) {
  const span = 1 << LEN_EXTRA[s];
  for (let k = 0; k < span && LEN_BASE[s] + k <= 258; k++) {
    LEN_SYM[LEN_BASE[s] + k] = 257 + s;
    LEN_EXT[LEN_BASE[s] + k] = k;
  }
}
const DIST_SYM = new Uint16Array(32769);
const DIST_EXT = new Uint16Array(32769);
for (let s = 0; s < 30; s++) {
  const span = 1 << DIST_EXTRA[s];
  for (let k = 0; k < span && DIST_BASE[s] + k <= 32768; k++) {
    DIST_SYM[DIST_BASE[s] + k] = s;
    DIST_EXT[DIST_BASE[s] + k] = k;
  }
}
const FIXED_LIT_LEN = new Uint8Array(288);
for (let i = 0; i < 288; i++) FIXED_LIT_LEN[i] = i < 144 ? 8 : i < 256 ? 9 : i < 280 ? 7 : 8;

// ---------- bit writer ----------
class BitWriter {
  constructor() {
    this.bytes = [];
    this.acc = 0;
    this.nbits = 0;
  }
  write(value, count) {
    this.acc |= value << this.nbits;
    this.nbits += count;
    while (this.nbits >= 8) {
      this.bytes.push(this.acc & 0xff);
      this.acc >>>= 8;
      this.nbits -= 8;
    }
  }
  writeReversed(code, count) {
    for (let i = count - 1; i >= 0; i--) this.write((code >>> i) & 1, 1);
  }
  bitLength() {
    return this.bytes.length * 8 + this.nbits;
  }
  finish() {
    if (this.nbits > 0) this.bytes.push(this.acc & 0xff);
    return Uint8Array.from(this.bytes);
  }
}

// ---------- length-limited Huffman ----------
function huffmanLengths(freqs, maxLen) {
  const n = freqs.length;
  const used = [];
  for (let i = 0; i < n; i++) if (freqs[i] > 0) used.push(i);
  const lengths = new Uint8Array(n);
  if (used.length === 0) return lengths;
  if (used.length === 1) {
    lengths[used[0]] = 1;
    return lengths;
  }
  // Unlimited Huffman via two queues (leaves sorted by frequency, internal nodes
  // in creation order, both nondecreasing).
  const leafQ = [...used].sort((a, b) => freqs[a] - freqs[b] || a - b);
  const intQ = [];
  const freqOf = new Float64Array(2 * n + 5);
  const left = new Int32Array(2 * n + 5);
  const right = new Int32Array(2 * n + 5);
  const isLeaf = new Uint8Array(2 * n + 5);
  for (const s of used) {
    freqOf[s] = freqs[s];
    isLeaf[s] = 1;
  }
  let p1 = 0;
  let p2 = 0;
  const takeNode = () => {
    if (p2 >= intQ.length) return leafQ[p1++];
    if (p1 >= leafQ.length) return intQ[p2++];
    return freqOf[leafQ[p1]] <= freqOf[intQ[p2]] ? leafQ[p1++] : intQ[p2++];
  };
  let nextId = n;
  let root = -1;
  for (let k = 0; k < used.length - 1; k++) {
    const a = takeNode();
    const b = takeNode();
    const id = nextId++;
    left[id] = a;
    right[id] = b;
    freqOf[id] = freqOf[a] + freqOf[b];
    intQ.push(id);
    root = id;
  }
  const bl = new Uint8Array(n);
  const stack = [[root, 0]];
  while (stack.length) {
    const [id, d] = stack.pop();
    if (isLeaf[id]) bl[id] = d;
    else {
      stack.push([left[id], d + 1]);
      stack.push([right[id], d + 1]);
    }
  }
  // Length counts; codes longer than maxLen are clamped into blCount[maxLen].
  const blCount = new Int32Array(maxLen + 1);
  let overflow = 0;
  for (const s of used) {
    if (bl[s] > maxLen) {
      overflow++;
      blCount[maxLen]++;
    } else blCount[bl[s]]++;
  }
  while (overflow > 0) {
    let bits = maxLen - 1;
    while (bits > 0 && blCount[bits] === 0) bits--;
    if (bits === 0) break;
    blCount[bits]--;
    blCount[bits + 1] += 2;
    blCount[maxLen]--;
    overflow = Math.max(0, overflow - 2);
  }
  // Kraft safety net: sum blCount[l] * 2^(maxLen-l) must be <= 2^maxLen.
  const kraftLimit = 1 << maxLen;
  const kraft = () => {
    let s = 0;
    for (let l = 1; l <= maxLen; l++) s += blCount[l] * (1 << (maxLen - l));
    return s;
  };
  while (kraft() > kraftLimit) {
    let l = maxLen - 1;
    while (l > 0 && blCount[l] === 0) l--;
    if (l === 0) break;
    blCount[l]--;
    blCount[l + 1]++;
  }
  // Reassign lengths by frequency: shortest to most frequent.
  const byFreqDesc = [...used].sort((a, b) => freqs[b] - freqs[a] || a - b);
  let k = 0;
  for (let len = 1; len <= maxLen; len++) {
    for (let j = 0; j < blCount[len]; j++) lengths[byFreqDesc[k++]] = len;
  }
  return lengths;
}

function canonicalCodes(lengths) {
  let maxLen = 0;
  for (const l of lengths) if (l > maxLen) maxLen = l;
  const blCount = new Int32Array(maxLen + 1);
  for (const l of lengths) if (l > 0) blCount[l]++;
  const nextCode = new Int32Array(maxLen + 1);
  let code = 0;
  for (let l = 1; l <= maxLen; l++) {
    code = (code + blCount[l - 1]) << 1;
    nextCode[l] = code;
  }
  const codes = new Uint32Array(lengths.length);
  for (let i = 0; i < lengths.length; i++) {
    const l = lengths[i];
    if (l > 0) codes[i] = nextCode[l]++;
  }
  return codes;
}

// ---------- match finder ----------
const HASH_BITS = 17;
const HASH_SIZE = 1 << HASH_BITS;
const HASH_MASK = HASH_SIZE - 1;
const MAX_MATCH = 258;
const WINDOW = 32768;

function hash3(d, i) {
  let h = ((d[i] << 16) ^ (d[i + 1] << 8) ^ d[i + 2]) >>> 0;
  h = Math.imul(h, 2654435761) >>> 0;
  return (h >>> (32 - HASH_BITS)) & HASH_MASK;
}
function hash4(d, i) {
  let h = ((d[i] << 24) ^ (d[i + 1] << 16) ^ (d[i + 2] << 8) ^ d[i + 3]) >>> 0;
  h = Math.imul(h, 2654435761) >>> 0;
  return (h >>> (32 - HASH_BITS)) & HASH_MASK;
}

class Matcher {
  constructor(data, maxChain = 128, goodLen = 64, hashBytes = 3, stopLen = 0, stopChain = 64, seedCap = 16, enableSeed = true) {
    this.d = data;
    this.n = data.length;
    this.maxChain = maxChain;
    this.goodLen = goodLen;
    this.hashBytes = hashBytes;
    this.stopLen = stopLen;
    this.stopChain = stopChain;
    this.seedCap = seedCap;
    this.enableSeed = enableSeed;
    this.head = new Int32Array(HASH_SIZE).fill(-1);
    this.prev = new Int32Array(this.n).fill(-1);
    this.inserted = 0;
    this.mLen = new Int16Array(this.n).fill(-1);
    this.mDist = new Int32Array(this.n);
    this.mLen2 = new Int16Array(this.n).fill(-1);
    this.mDist2 = new Int32Array(this.n);
  }
  insertUpTo(pos) {
    const d = this.d;
    const need = this.hashBytes === 4 ? 3 : 2;
    while (this.inserted < pos && this.inserted + need < this.n) {
      const h = this.hashBytes === 4 ? hash4(d, this.inserted) : hash3(d, this.inserted);
      this.prev[this.inserted] = this.head[h];
      this.head[h] = this.inserted;
      this.inserted++;
    }
  }
  find(i) {
    const d = this.d;
    const n = this.n;
    if (i + 3 > n) return { len: 0, dist: 0, len2: 0, dist2: 0 };
    this.insertUpTo(i);
    const maxLen = Math.min(MAX_MATCH, n - i);
    const h = this.hashBytes === 4 ? hash4(d, i) : hash3(d, i);
    let cand = this.head[h];
    let bestLen = 0;
    let bestDist = 0;
    let len2 = 0;
    let dist2 = 0;
    // Seed from the previous position's match, shifted by one byte: a match of
    // length L at distance d starting at i-1 implies one of length L-1 at the
    // same distance starting at i (and it may run longer; extend it directly).
    // The seed is kept OUT of the chain walk's quick-reject: the walk measures
    // near candidates with its own bestLen (starting at 0), and the seed is
    // merged afterwards. (Feeding the seed into the walk's bestLen made the
    // quick-reject discard near matches before measuring them: -9kB on i007.)
    // When the seed is already long, the walk is capped to the most recent
    // candidates; the deep walk is reserved for positions with no seed.
    let seedLen = 0;
    let seedDist = 0;
    let chainBudget = this.maxChain;
    if (this.enableSeed && i > 0 && this.mLen[i - 1] >= 4) {
      const d0 = this.mDist[i - 1];
      let len = this.mLen[i - 1] - 1;
      while (len < maxLen && d[i - d0 + len] === d[i + len]) len++;
      seedLen = len;
      seedDist = d0;
      if (this.mLen2[i - 1] >= 4) {
        len2 = this.mLen2[i - 1] - 1;
        dist2 = this.mDist2[i - 1];
      }
      if (seedLen >= 32) chainBudget = this.seedCap;
    }
    let chain = 0;
    const limit = i - WINDOW;
    while (cand >= limit && cand >= 0 && chain < chainBudget) {
      chain++;
      if (bestLen > 0 && d[cand + bestLen] !== d[i + bestLen]) {
        cand = this.prev[cand];
        continue;
      }
      let len = 0;
      while (len < maxLen && d[cand + len] === d[i + len]) len++;
      if (len > bestLen) {
        bestLen = len;
        bestDist = i - cand;
        if (len >= maxLen) break;
        // A good match only shortens the remaining search, it never ends it:
        // a nearer candidate may still beat it on distance cost.
        if (len >= this.goodLen && chain > 32) break;
      }
      if (len >= 3 && dist2 === 0) {
        len2 = len;
        dist2 = i - cand;
      }
      // Stop rule: once a decent match is in hand, deep chains rarely pay.
      if (this.stopLen > 0 && bestLen >= this.stopLen && chain > this.stopChain) break;
      cand = this.prev[cand];
    }
    if (seedLen > bestLen) {
      if (bestLen >= 3 && dist2 === 0) {
        len2 = bestLen;
        dist2 = bestDist;
      }
      bestLen = seedLen;
      bestDist = seedDist;
    }
    return { len: bestLen, dist: bestDist, len2, dist2 };
  }
}

// ---------- optimal parse ----------
function optimalParse(data, matcher, litCost, lenCost, distCost) {
  const n = data.length;
  const cost = new Float64Array(n + 1).fill(Infinity);
  const from = new Int32Array(n + 1);
  const tlen = new Uint16Array(n + 1);
  const tdist = new Int32Array(n + 1);
  cost[0] = 0;
  const mLen = matcher.mLen;
  const mDist = matcher.mDist;
  const mLen2 = matcher.mLen2;
  const mDist2 = matcher.mDist2;
  // Precompute, once per parse, the length-symbol transitions for every possible
  // match length: for a match of length ml, the tried landing lengths L and the
  // symbol bit cost (excluding the distance part, which varies per match).
  // This moves the LEN_BASE/LEN_EXTRA/Math.min work out of the hot loop.
  const transL = new Array(259);
  const transC = new Array(259);
  for (let ml = 3; ml <= 258; ml++) {
    const ls = [];
    const cs = [];
    for (let s = 0; s < 29; s++) {
      const base = LEN_BASE[s];
      if (base > ml) break;
      ls.push(Math.min(base + ((1 << LEN_EXTRA[s]) - 1), ml));
      cs.push(lenCost[257 + s] + LEN_EXTRA[s]);
    }
    transL[ml] = Int16Array.from(ls);
    transC[ml] = Float64Array.from(cs);
  }
  for (let i = 0; i < n; i++) {
    const ci = cost[i];
    if (ci === Infinity) continue;
    const lc = ci + litCost[data[i]];
    if (lc < cost[i + 1]) {
      cost[i + 1] = lc;
      from[i + 1] = i;
      tlen[i + 1] = 1;
      tdist[i + 1] = 0;
    }
    if (i + 3 <= n) {
      if (mLen[i] === -1) {
        const m = matcher.find(i);
        mLen[i] = m.len;
        mDist[i] = m.dist;
        mLen2[i] = m.len2;
        mDist2[i] = m.dist2;
      }
      for (let t = 0; t < 2; t++) {
        const ml = t === 0 ? mLen[i] : mLen2[i];
        const md = t === 0 ? mDist[i] : mDist2[i];
        if (ml < 3 || md <= 0) continue;
        if (t === 1 && md === mDist[i]) continue;
        const ds = DIST_SYM[md];
        const dc = distCost[ds] + DIST_EXTRA[ds];
        const ls = transL[ml];
        const cs = transC[ml];
        for (let s = 0; s < ls.length; s++) {
          const L = ls[s];
          const c = ci + cs[s] + dc;
          if (c < cost[i + L]) {
            cost[i + L] = c;
            from[i + L] = i;
            tlen[i + L] = L;
            tdist[i + L] = md;
          }
        }
      }
    }
  }
  const tokens = [];
  let j = n;
  while (j > 0) {
    const i = from[j];
    tokens.push({ len: tlen[j], dist: tdist[j], lit: tdist[j] === 0 ? data[i] : -1 });
    j = i;
  }
  tokens.reverse();
  return tokens;
}

// ---------- emit one dynamic block ----------
const CL_ORDER = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];

function emitDynamicBlock(w, tokens, litLen, litCode, distLen, distCode, finalBlock = true) {
  let hlit = 257;
  for (let i = 286; i >= 257; i--) if (litLen[i] > 0) { hlit = i + 1; break; }
  let hdist = 1;
  for (let i = 29; i >= 0; i--) if (distLen[i] > 0) { hdist = i + 1; break; }
  const all = [];
  for (let i = 0; i < hlit; i++) all.push(litLen[i]);
  for (let i = 0; i < hdist; i++) all.push(distLen[i]);
  const seq = [];
  let i = 0;
  while (i < all.length) {
    const v = all[i];
    let run = 1;
    while (i + run < all.length && all[i + run] === v) run++;
    if (v === 0) {
      let r = run;
      while (r >= 11) {
        const take = Math.min(r, 138);
        seq.push([18, take - 11, 7]);
        r -= take;
      }
      if (r >= 3) {
        seq.push([17, r - 3, 3]);
        r = 0;
      }
      while (r > 0) {
        seq.push([0, 0, 0]);
        r--;
      }
    } else {
      seq.push([v, 0, 0]);
      let r = run - 1;
      while (r >= 3) {
        const take = Math.min(r, 6);
        seq.push([16, take - 3, 2]);
        r -= take;
      }
      while (r > 0) {
        seq.push([v, 0, 0]);
        r--;
      }
    }
    i += run;
  }
  const clFreq = new Int32Array(19);
  for (const [s] of seq) clFreq[s]++;
  const clLen = huffmanLengths(clFreq, 7);
  const clCode = canonicalCodes(clLen);
  let hclen = 19;
  while (hclen > 4 && clLen[CL_ORDER[hclen - 1]] === 0) hclen--;
  w.write(finalBlock ? 1 : 0, 1); // BFINAL
  w.write(2, 2); // BTYPE = dynamic
  w.write(hlit - 257, 5);
  w.write(hdist - 1, 5);
  w.write(hclen - 4, 4);
  for (let k = 0; k < hclen; k++) w.write(clLen[CL_ORDER[k]], 3);
  for (const [s, extra, bits] of seq) {
    w.writeReversed(clCode[s], clLen[s]);
    if (bits > 0) w.write(extra, bits);
  }
  for (const t of tokens) {
    if (t.dist === 0) {
      w.writeReversed(litCode[t.lit], litLen[t.lit]);
    } else {
      const ls = LEN_SYM[t.len];
      w.writeReversed(litCode[ls], litLen[ls]);
      if (LEN_EXTRA[ls - 257] > 0) w.write(LEN_EXT[t.len], LEN_EXTRA[ls - 257]);
      const ds = DIST_SYM[t.dist];
      w.writeReversed(distCode[ds], distLen[ds]);
      if (DIST_EXTRA[ds] > 0) w.write(DIST_EXT[t.dist], DIST_EXTRA[ds]);
    }
  }
  w.writeReversed(litCode[256], litLen[256]);
}

// ---------- adler32 ----------
function adler32(data) {
  let a = 1;
  let b = 0;
  for (let i = 0; i < data.length;) {
    const end = Math.min(data.length, i + 5552);
    for (; i < end; i++) {
      a += data[i];
      b += a;
    }
    a %= 65521;
    b %= 65521;
  }
  return ((b << 16) | a) >>> 0;
}

/**
 * One dynamic block for `data`, written into the shared bit writer `w`
 * (blocks are bit-packed consecutively; only the final block's writer is padded).
 * Iterates parse/re-cost and keeps the smallest emit by exact bit count.
 */
function deflateBlock(data, opts, finalBlock, w) {
  const matcher = new Matcher(data, opts.maxChain, opts.goodLen, opts.hashBytes, opts.stopLen, opts.stopChain, opts.seedCap, opts.enableSeed);
  const fixedDist = new Uint8Array(30).fill(5);
  let litCost = FIXED_LIT_LEN;
  let lenCost = FIXED_LIT_LEN;
  let distCost = fixedDist;
  let best = null;
  for (let it = 0; it < opts.iterations; it++) {
    const tokens = optimalParse(data, matcher, litCost, lenCost, distCost);
    const litFreq = new Int32Array(288);
    const distFreq = new Int32Array(30);
    for (const t of tokens) {
      if (t.dist === 0) litFreq[t.lit]++;
      else {
        litFreq[LEN_SYM[t.len]]++;
        distFreq[DIST_SYM[t.dist]]++;
      }
    }
    litFreq[256] = 1;
    const litLen = huffmanLengths(litFreq, 15);
    let distLen = huffmanLengths(distFreq, 15);
    if (distFreq.every((x) => x === 0)) distLen[0] = 1;
    const trial = new BitWriter();
    const litCode = canonicalCodes(litLen);
    const distCode = canonicalCodes(distLen);
    emitDynamicBlock(trial, tokens, litLen, litCode, distLen, distCode, finalBlock);
    const bits = trial.bitLength();
    if (best === null || bits < best.bits) best = { bits, tokens, litLen, distLen };
    litCost = litLen;
    lenCost = litLen;
    distCost = distLen;
  }
  const litCode = canonicalCodes(best.litLen);
  const distCode = canonicalCodes(best.distLen);
  emitDynamicBlock(w, best.tokens, best.litLen, litCode, best.distLen, distCode, finalBlock);
}

/**
 * Deflate `data` with optimal parsing; returns a complete zlib stream.
 * opts: blockBytes (0 = one block; rounded down to a multiple of rowLen),
 * rowLen (bytes per filtered row, for block alignment), maxChain (default 128),
 * goodLen (default 64), iterations (default 2), hashBytes (3 or 4, default 3),
 * stopLen/stopChain (early chain stop once a match of stopLen is found),
 * seedCap (chain budget when a long seed match exists, default 16),
 * enableSeed (seed matches from the previous position, default true).
 */
export function deflateOptimal(data, opts = {}) {
  const o = {
    blockBytes: opts.blockBytes ?? 0,
    rowLen: opts.rowLen ?? 0,
    maxChain: opts.maxChain ?? 128,
    goodLen: opts.goodLen ?? 64,
    iterations: opts.iterations ?? 2,
    hashBytes: opts.hashBytes ?? 3,
    stopLen: opts.stopLen ?? 0,
    stopChain: opts.stopChain ?? 64,
    seedCap: opts.seedCap ?? 16,
    enableSeed: opts.enableSeed ?? true,
  };
  let blockBytes = o.blockBytes;
  if (blockBytes > 0 && o.rowLen > 0) blockBytes = Math.max(o.rowLen, Math.floor(blockBytes / o.rowLen) * o.rowLen);
  const w = new BitWriter();
  if (blockBytes <= 0 || blockBytes >= data.length) {
    deflateBlock(data, o, true, w);
  } else {
    for (let at = 0; at < data.length; at += blockBytes) {
      const end = Math.min(data.length, at + blockBytes);
      deflateBlock(data.subarray(at, end), o, end === data.length, w);
    }
  }
  const body = w.finish();
  const out = new Uint8Array(2 + body.length + 4);
  out[0] = 0x78;
  out[1] = 0x9c;
  out.set(body, 2);
  const ad = adler32(data);
  out[2 + body.length] = (ad >>> 24) & 0xff;
  out[2 + body.length + 1] = (ad >>> 16) & 0xff;
  out[2 + body.length + 2] = (ad >>> 8) & 0xff;
  out[2 + body.length + 3] = ad & 0xff;
  return out;
}

export { inflateSync };
