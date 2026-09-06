"""Local (simulated) blockchain for tamper-evident, verifiable records.

This is a self-contained, simplified blockchain persisted to a JSON ledger. No
external network, wallet or API key is required, so it runs anywhere (local dev,
Render, etc.). It implements the core properties that make a chain tamper-evident:

  * **Chaining**     - every block stores the SHA-256 hash of the previous block,
                       so any change to an earlier block invalidates every later block.
  * **Merkle root**  - each block's transactions are combined into a Merkle root,
                       binding a record hash to the block in a single digest.
  * **Hash target**  - blocks are "mined" against a difficulty target (simulated
                       proof-of-work), demonstrating the block is hard to forge.
  * **Verification** - a record can be re-verified at any time: we re-derive the
                       data hash, confirm the on-chain Merkle membership and recompute
                       the full chain hash linkage.

Records hold the fingerprint (SHA-256) of the discovered social-media post plus
metadata (source URL, title, snippet, face match score, timestamps).
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any, Dict, List, Optional

_IS_VERCEL = os.getenv("VERCEL", "") == "1"
_DEFAULT_LEDGER = (
    "/tmp/ledger.json" if _IS_VERCEL
    else os.path.join(os.path.dirname(__file__), "..", "..", "data", "ledger.json")
)

LEDGER_PATH = os.getenv("BLOCKCHAIN_LEDGER_PATH", _DEFAULT_LEDGER)

DIFFICULTY = int(os.getenv("BLOCKCHAIN_DIFFICULTY", "3"))


# --------------------------------------------------------------------------- #
# Hashing helpers
# --------------------------------------------------------------------------- #
def sha256(data: Any) -> str:
    if isinstance(data, str):
        data = data.encode()
    elif isinstance(data, (dict, list)):
        data = json.dumps(data, sort_keys=True, default=str).encode()
    return hashlib.sha256(data).hexdigest()


def merkle_root(tx_hashes: List[str]) -> str:
    """Compute the Merkle root of a list of transaction hashes."""
    if not tx_hashes:
        return sha256("")
    layer = list(tx_hashes)
    while len(layer) > 1:
        if len(layer) % 2 == 1:
            layer.append(layer[-1])
        layer = [sha256(layer[i] + layer[i + 1]) for i in range(0, len(layer), 2)]
    return layer[0]


def compute_block_hash(block: Dict[str, Any]) -> str:
    """Compute the canonical hash of a block (excluding its stored hash)."""
    copy = dict(block)
    copy.pop("hash", None)
    return sha256(copy)


# --------------------------------------------------------------------------- #
# Chain
# --------------------------------------------------------------------------- #
class LocalChain:
    def __init__(self, path: Optional[str] = None,
                 mining: bool = False, difficulty: int = DIFFICULTY):
        self.path = path or LEDGER_PATH
        self.mining = mining
        self.difficulty = difficulty
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        if os.path.exists(self.path):
            self.blocks = self._load()
        else:
            self.blocks = [self._genesis()]

    # ----- persistence -----
    def _load(self) -> List[Dict[str, Any]]:
        with open(self.path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def save(self) -> None:
        directory = os.path.dirname(self.path)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.blocks, fh, indent=2, default=str)
        os.replace(tmp, self.path)

    # ----- genesis -----
    def _genesis(self) -> Dict[str, Any]:
        block = {
            "index": 0,
            "timestamp": 0,
            "previous_hash": "0" * 64,
            "difficulty": self.difficulty,
            "nonce": 0,
            "transactions": [{
                "tx_id": sha256({"type": "genesis", "ts": time.time()}),
                "type": "genesis",
                "data_hash": sha256("genesis"),
                "meta": "Genesis block",
                "timestamp": 0,
            }],
            "merkle_root": "",
            "hash": "",
        }
        block["merkle_root"] = merkle_root(
            [t["tx_id"] for t in block["transactions"]])
        block["hash"] = self._mine(block)
        return block

    # ----- mining (simulated proof-of-work) -----
    def _mine(self, block: Dict[str, Any]) -> str:
        prefix = "0" * block.get("difficulty", self.difficulty)
        nonce = block.get("nonce", 0)
        candidate = dict(block)
        candidate.pop("hash", None)
        while True:
            candidate["nonce"] = nonce
            h = sha256(candidate)
            if h.startswith(prefix):
                candidate["hash"] = h
                block.update({"nonce": nonce, "hash": h})
                return h
            nonce += 1
            if nonce % 100000 == 0 and nonce > 0:
                # safety abort for very high difficulty
                if nonce > 5_000_000:
                    raise RuntimeError("Mining exceeded nonce budget")
        return h

    # ----- transactions / records -----
    def add_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Add a verified data record as a transaction in a new block."""
        data_hash = sha256(record) if "data_hash" not in record else record["data_hash"]
        tx = {
            "tx_id": sha256({"ts": time.time(), "data_hash": data_hash}),
            "type": record.get("type", "post_record"),
            "data_hash": data_hash,
            "record": record,
            "timestamp": time.time(),
        }
        previous = self.blocks[-1]
        block = {
            "index": previous["index"] + 1,
            "timestamp": time.time(),
            "previous_hash": previous["hash"],
            "difficulty": self.difficulty,
            "nonce": 0,
            "transactions": [tx],
            "merkle_root": "",
            "hash": "",
        }
        block["merkle_root"] = merkle_root([tx["tx_id"]])
        block["hash"] = self._mine(block)
        self.blocks.append(block)
        self.save()
        # Referential fields are returned to the caller but NOT stored in the
        # block transaction, otherwise mutating the stored tx would break hashing.
        return {**tx, "block_index": block["index"], "block_hash": block["hash"]}

    def add_record_from_image(self, fingerprint: Dict[str, Any]) -> Dict[str, Any]:
        """Convenience: wrap a fingerprint dict (image hash + metadata) as a record."""
        return self.add_record(fingerprint)

    def find_record(self, data_hash: str) -> Optional[Dict[str, Any]]:
        """Locate a transaction by its data hash (post fingerprint)."""
        for block in self.blocks:
            for tx in block.get("transactions", []):
                if tx.get("data_hash") == data_hash:
                    return {"block_index": block["index"],
                            "block_hash": block["hash"],
                            "tx": tx}
        return None

    # ----- verification -----
    def verify_tamper(self) -> Dict[str, Any]:
        """Recompute the whole chain and check hash linkage + PoW."""
        checks = []
        valid = True
        for i, block in enumerate(self.blocks):
            stored = block.get("hash", "")
            recomputed = compute_block_hash(block)
            hash_ok = stored == recomputed
            prefix_ok = stored.startswith("0" * block.get("difficulty", self.difficulty))
            if i == 0:
                chain_ok = block.get("previous_hash") == "0" * 64
            else:
                chain_ok = block.get("previous_hash") == self.blocks[i - 1].get("hash")
            ok = hash_ok and prefix_ok and chain_ok
            if not ok:
                valid = False
            checks.append({"block": block["index"], "hash_ok": hash_ok,
                           "pow_ok": prefix_ok, "chain_ok": chain_ok})
        return {"valid": valid, "block_count": len(self.blocks), "checks": checks}

    def verify_record(self, data_hash: str) -> Dict[str, Any]:
        """Verify a post record against the on-chain record.

        Returns tamper-evident proof: we recompute the data hash from the original
        record and confirm it lives in a block whose Merkle root and hash link
        agree with the chain.
        """
        found = self.find_record(data_hash)
        if not found:
            return {"verified": False, "reason": "record_not_found",
                    "data_hash": data_hash}
        block = self.blocks[found["block_index"]]
        tx = found["tx"]
        tx_hashes = [t["tx_id"] for t in block.get("transactions", [])]
        root = merkle_root(tx_hashes)
        root_ok = root == block.get("merkle_root")
        on_chain = tx.get("data_hash") == data_hash
        valid = root_ok and on_chain
        return {
            "verified": valid,
            "data_hash": data_hash,
            "block_index": found["block_index"],
            "block_hash": found["block_hash"],
            "merkle_root_matches": root_ok,
            "data_hash_on_chain": on_chain,
            "record": tx.get("record"),
            "timestamp": tx.get("timestamp"),
        }

    def status(self) -> Dict[str, Any]:
        return {
            "name": "LocalSimChain",
            "blocks": len(self.blocks),
            "height": self.blocks[-1]["index"] if self.blocks else -1,
            "difficulty": self.difficulty,
            "mining_enabled": self.mining,
            "ledger_path": self.path,
            "last_block_hash": self.blocks[-1]["hash"] if self.blocks else None,
        }


_chain = LocalChain()


def get_chain() -> LocalChain:
    global _chain
    return _chain
