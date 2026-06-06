"""Tests for per-companion bridge settings parsing and startup guard."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from repeater.companion.utils import (
    COMPANION_SETTINGS_ALLOWLIST,
    CompanionContactCapacityError,
    check_companion_contact_capacity,
    effective_max_contacts,
    enforce_companion_contact_capacity,
    merge_companion_settings_update,
    parse_companion_bridge_kwargs,
    parse_positive_int,
    select_companion_contacts_to_trim,
    trim_companion_contacts_to_fit,
    validate_companion_config_capacity,
)

# pymc_core defaults (CompanionBridge / ContactStore)
_DEFAULT_MAX_CONTACTS = 1000


class TestParsePositiveInt:
    def test_valid(self):
        assert parse_positive_int("100", "max_contacts") == 100

    def test_invalid_type(self):
        with pytest.raises(ValueError, match="max_contacts"):
            parse_positive_int("abc", "max_contacts")

    def test_below_minimum(self):
        with pytest.raises(ValueError, match="max_contacts"):
            parse_positive_int(0, "max_contacts")


class TestParseCompanionBridgeKwargs:
    def test_empty_settings(self):
        assert parse_companion_bridge_kwargs({}) == {}

    def test_max_contacts_and_offline_queue(self):
        assert parse_companion_bridge_kwargs(
            {"max_contacts": 2000, "offline_queue_size": 1024}
        ) == {"max_contacts": 2000, "offline_queue_size": 1024}

    def test_ignored_keys_warn(self, caplog):
        caplog.set_level(logging.WARNING)
        result = parse_companion_bridge_kwargs(
            {"max_contacts": 500, "max_channels": 64, "adv_type": 2}
        )
        assert result == {"max_contacts": 500}
        assert any("max_channels" in r.message for r in caplog.records)
        assert any("adv_type" in r.message for r in caplog.records)

    def test_invalid_max_contacts(self):
        with pytest.raises(ValueError):
            parse_companion_bridge_kwargs({"max_contacts": -1})


class TestEffectiveMaxContacts:
    def test_default(self):
        assert effective_max_contacts({}) == _DEFAULT_MAX_CONTACTS

    def test_override(self):
        assert effective_max_contacts({"max_contacts": 500}) == 500


class TestMergeCompanionSettingsUpdate:
    def test_merges_bridge_settings(self):
        merged = merge_companion_settings_update(
            {"node_name": "a"},
            {"max_contacts": 500},
        )
        assert merged == {"node_name": "a", "max_contacts": 500}

    def test_unknown_key_raises(self):
        with pytest.raises(ValueError, match="Unknown companion setting"):
            merge_companion_settings_update({}, {"max_channels": 64})


class TestValidateCompanionConfigCapacity:
    def test_uses_merged_settings_not_stale_identity(self):
        identity = {
            "identity_key": "aa" * 32,
            "settings": {"max_contacts": 1000},
        }
        sqlite = MagicMock()
        sqlite.companion_count_contacts.return_value = 600
        with pytest.raises(CompanionContactCapacityError):
            validate_companion_config_capacity(
                identity,
                sqlite,
                settings={"max_contacts": 500},
            )
        sqlite.companion_count_contacts.assert_called_once()


class TestCheckCompanionContactCapacity:
    def test_skips_without_sqlite(self):
        check_companion_contact_capacity("0x01", 100, None)

    def test_passes_when_under_limit(self):
        sqlite = MagicMock()
        sqlite.companion_count_contacts.return_value = 100
        check_companion_contact_capacity("0x01", 500, sqlite)

    def test_raises_when_over_limit(self):
        sqlite = MagicMock()
        sqlite.companion_count_contacts.return_value = 812
        with pytest.raises(CompanionContactCapacityError) as exc:
            check_companion_contact_capacity("0xab", 500, sqlite, companion_name="BotCompanion")
        assert exc.value.stored_count == 812
        assert exc.value.max_contacts == 500
        assert "BotCompanion" in str(exc.value)


class TestOfflineQueueOff:
    def test_zero_allowed(self):
        assert parse_companion_bridge_kwargs({"offline_queue_size": 0}) == {"offline_queue_size": 0}

    def test_max_contacts_zero_still_rejected(self):
        with pytest.raises(ValueError, match="max_contacts"):
            parse_companion_bridge_kwargs({"max_contacts": 0})


class TestSelectCompanionContactsToTrim:
    @staticmethod
    def _c(pk, flags=0, lastmod=0):
        return {"pubkey": pk, "flags": flags, "lastmod": lastmod}

    def test_under_limit_keeps_all(self):
        contacts = [self._c(b"\x01"), self._c(b"\x02")]
        keep, removed = select_companion_contacts_to_trim(contacts, 5)
        assert removed == []
        assert keep == contacts

    def test_evicts_oldest_non_favourite_and_protects_favourites(self):
        contacts = [
            self._c(b"\x01", lastmod=10),
            self._c(b"\x02", lastmod=30),
            self._c(b"\x03", flags=1, lastmod=5),  # favourite + oldest -> protected
            self._c(b"\x04", lastmod=20),
        ]
        keep, removed = select_companion_contacts_to_trim(contacts, 2)
        assert {c["pubkey"] for c in keep} == {b"\x03", b"\x02"}
        assert {c["pubkey"] for c in removed} == {b"\x01", b"\x04"}

    def test_refuses_when_favourites_exceed_limit(self):
        contacts = [
            self._c(b"\x01", flags=1, lastmod=1),
            self._c(b"\x02", flags=1, lastmod=2),
        ]
        with pytest.raises(ValueError, match="favourite"):
            select_companion_contacts_to_trim(contacts, 1)


class TestSqliteRetentionTrim:
    @staticmethod
    def _handler(tmp_path):
        from repeater.data_acquisition.sqlite_handler import SQLiteHandler

        return SQLiteHandler(tmp_path)

    @staticmethod
    def _push(h, companion_hash, i, max_messages=None):
        return h.companion_push_message(
            companion_hash,
            {"text": f"m{i}", "timestamp": i, "packet_hash": f"{companion_hash}-{i}"},
            max_messages=max_messages,
        )

    def test_trims_to_max_messages(self, tmp_path):
        h = self._handler(tmp_path)
        for i in range(5):
            self._push(h, "0x01", i, max_messages=3)
        assert len(h.companion_load_messages("0x01")) == 3

    def test_none_keeps_all(self, tmp_path):
        h = self._handler(tmp_path)
        for i in range(5):
            self._push(h, "0x01", i, max_messages=None)
        assert len(h.companion_load_messages("0x01")) == 5

    def test_trim_isolated_per_companion(self, tmp_path):
        h = self._handler(tmp_path)
        for i in range(4):
            self._push(h, "0x01", i, max_messages=2)
        for i in range(3):
            self._push(h, "0x02", i, max_messages=None)
        assert len(h.companion_load_messages("0x01")) == 2
        assert len(h.companion_load_messages("0x02")) == 3


class TestTrimContactsOnOverflowPolicy:
    @staticmethod
    def _contacts(n, favourites=0):
        out = []
        for i in range(n):
            flags = 1 if i < favourites else 0
            out.append({"pubkey": i.to_bytes(2, "big"), "flags": flags, "lastmod": i})
        return out

    def test_allowlist_includes_policy_key(self):
        assert "trim_contacts_on_overflow" in COMPANION_SETTINGS_ALLOWLIST
        # And it is accepted by the settings merge.
        merged = merge_companion_settings_update({}, {"trim_contacts_on_overflow": True})
        assert merged == {"trim_contacts_on_overflow": True}

    def test_trim_helper_persists_kept_set(self):
        sqlite = MagicMock()
        sqlite.companion_load_contacts.return_value = self._contacts(5)
        sqlite.companion_save_contacts.return_value = True
        removed = trim_companion_contacts_to_fit(sqlite, "0x01", 3)
        assert removed == 2
        saved_hash, saved_contacts = sqlite.companion_save_contacts.call_args[0]
        assert saved_hash == "0x01"
        assert len(saved_contacts) == 3

    def test_trim_helper_noop_when_under_limit(self):
        sqlite = MagicMock()
        sqlite.companion_load_contacts.return_value = self._contacts(2)
        assert trim_companion_contacts_to_fit(sqlite, "0x01", 5) == 0
        sqlite.companion_save_contacts.assert_not_called()

    def test_enforce_guards_by_default(self):
        sqlite = MagicMock()
        sqlite.companion_count_contacts.return_value = 600
        with pytest.raises(CompanionContactCapacityError):
            enforce_companion_contact_capacity("0x01", 500, sqlite)
        sqlite.companion_save_contacts.assert_not_called()

    def test_enforce_trims_when_policy_enabled(self):
        sqlite = MagicMock()
        sqlite.companion_load_contacts.return_value = self._contacts(600)
        sqlite.companion_save_contacts.return_value = True
        removed = enforce_companion_contact_capacity("0x01", 500, sqlite, trim=True)
        assert removed == 100


class TestPersistSkipWhenOff:
    @staticmethod
    def _frame_server(max_size):
        from repeater.companion.frame_server import CompanionFrameServer

        fs = CompanionFrameServer.__new__(CompanionFrameServer)
        fs.sqlite_handler = MagicMock()
        fs.companion_hash = "0x01"
        bridge = MagicMock()
        bridge.message_queue._max_size = max_size
        fs.bridge = bridge
        return fs

    def test_skips_persistence_when_retention_zero(self):
        import asyncio

        fs = self._frame_server(0)
        asyncio.run(fs._persist_companion_message({"text": "x"}))
        fs.sqlite_handler.companion_push_message.assert_not_called()
        fs.bridge.message_queue.pop_last.assert_called_once()

    def test_persists_with_retention(self):
        import asyncio

        fs = self._frame_server(7)
        asyncio.run(fs._persist_companion_message({"text": "x"}))
        fs.sqlite_handler.companion_push_message.assert_called_once_with("0x01", {"text": "x"}, 7)
