"""Tests for the local blockchain implementation."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.blockchain.chain import LocalChain, merkle_root, sha256


def test_merkle_root_deterministic():
    items = [sha256(f"item {i}") for i in range(5)]
    r1 = merkle_root(items)
    r2 = merkle_root(items)
    assert r1 == r2
    assert len(r1) == 64


def test_genesis_and_add_record():
    with tempfile.TemporaryDirectory() as td:
        chain = LocalChain(path=os.path.join(td, "chain.json"), difficulty=2)
        assert chain.blocks[0]["index"] == 0
        record = {"type": "post_record", "post_url": "https://example.com/x",
                  "data": "hello"}
        tx = chain.add_record(record)
        assert tx["data_hash"] == sha256(record)
        assert chain.blocks[-1]["previous_hash"] == chain.blocks[0]["hash"]
        tamper = chain.verify_tamper()
        assert tamper["valid"] is True


def test_tamper_detection():
    with tempfile.TemporaryDirectory() as td:
        chain = LocalChain(path=os.path.join(td, "chain.json"), difficulty=2)
        record = {"type": "post_record", "post_url": "https://example.com/y"}
        tx = chain.add_record(record)
        assert chain.verify_tamper()["valid"] is True
        # tamper with an earlier block
        chain.blocks[1]["transactions"][0]["record"]["post_url"] = "https://evil.com"
        result = chain.verify_tamper()
        assert result["valid"] is False


def test_verify_record():
    with tempfile.TemporaryDirectory() as td:
        chain = LocalChain(path=os.path.join(td, "chain.json"), difficulty=2)
        record = {"type": "post_record", "post_url": "https://example.com/z"}
        tx = chain.add_record(record)
        v = chain.verify_record(tx["data_hash"])
        assert v["verified"] is True
        # different hash not found
        v2 = chain.verify_record("00" * 32)
        assert v2["verified"] is False


if __name__ == "__main__":
    for fn in [test_merkle_root_deterministic, test_genesis_and_add_record,
               test_tamper_detection, test_verify_record]:
        fn()
        print(f"ok  {fn.__name__}")
    print("all chain tests passed")