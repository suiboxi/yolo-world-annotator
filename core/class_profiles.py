"""Persistent, class-id-stable visual profiles.

Profiles are descriptive policy, never a source of class ids.  The GUI class
list remains authoritative; :func:`sync_profiles` reassigns ids in that exact
order whenever a project is opened or classes are edited.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from utils.config import as_bool, atomic_write_json, load_json


def _feature_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        return {"name": value, "required": False}
    if isinstance(value, Mapping):
        name = str(value.get("name", value.get("feature", ""))).strip()
        if not name:
            return {}
        return {"name": name, "required": as_bool(value.get("required", False))}
    return {}


@dataclass(slots=True)
class ClassProfile:
    class_name: str
    class_id: int
    yolo_prompt: str = ""
    siglip_prompt: str = ""
    vlm_description: str = ""
    always_vlm_verify: bool = False
    features: list[dict[str, Any]] = field(default_factory=list)
    required_features: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.class_name = str(self.class_name).strip()
        self.class_id = int(self.class_id)
        self.yolo_prompt = str(self.yolo_prompt or self.class_name)
        self.siglip_prompt = str(self.siglip_prompt or self.class_name)
        self.vlm_description = str(self.vlm_description or self.class_name)
        if isinstance(self.features, dict):
            normalized = [
                _feature_dict({"name": key, "required": value})
                for key, value in self.features.items()
            ]
        else:
            normalized = [_feature_dict(item) for item in self.features]
        self.features = [item for item in normalized if item]
        required = {str(item).strip() for item in self.required_features if str(item).strip()}
        required.update(item["name"] for item in self.features if item.get("required"))
        self.required_features = sorted(required)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, default_id: int = 0) -> "ClassProfile":
        raw_features = data.get("features", [])
        if isinstance(raw_features, Mapping):
            raw_features = [
                {"name": key, "required": value} for key, value in raw_features.items()
            ]
        return cls(
            class_name=str(data.get("class_name", data.get("name", ""))),
            class_id=int(data.get("class_id", default_id)),
            yolo_prompt=str(data.get("yolo_prompt", "")),
            siglip_prompt=str(data.get("siglip_prompt", "")),
            vlm_description=str(data.get("vlm_description", "")),
            always_vlm_verify=as_bool(data.get("always_vlm_verify", False)),
            features=list(raw_features) if isinstance(raw_features, list) else [],
            required_features=(
                list(data.get("required_features", []))
                if isinstance(data.get("required_features", []), list)
                else []
            ),
        )


class ClassProfiles:
    """Ordered registry keyed by both stable class id and class name."""

    def __init__(self, profiles: Iterable[ClassProfile] = ()) -> None:
        self._profiles: list[ClassProfile] = list(profiles)
        self._reindex()

    def _reindex(self) -> None:
        self._by_name = {profile.class_name: profile for profile in self._profiles if profile.class_name}
        self._by_id = {profile.class_id: profile for profile in self._profiles}

    def __iter__(self):
        return iter(self._profiles)

    def __len__(self) -> int:
        return len(self._profiles)

    def __getitem__(self, key: int | str) -> ClassProfile:
        profile = self.get(key)
        if profile is None:
            raise KeyError(key)
        return profile

    @property
    def profiles(self) -> list[ClassProfile]:
        return list(self._profiles)

    def get(self, key: int | str, default: ClassProfile | None = None) -> ClassProfile | None:
        if isinstance(key, int):
            return self._by_id.get(key, default)
        return self._by_name.get(str(key), default)

    def by_name(self, name: str) -> ClassProfile | None:
        return self.get(name)

    def by_id(self, class_id: int) -> ClassProfile | None:
        return self.get(int(class_id))

    def to_dict(self) -> dict[str, Any]:
        return {"version": 1, "profiles": [profile.to_dict() for profile in self._profiles]}

    def sync_classes(self, classes: Iterable[str]) -> "ClassProfiles":
        """Return a registry whose ids exactly follow *classes* order."""

        existing = self._by_name
        profiles: list[ClassProfile] = []
        for class_id, raw_name in enumerate(classes):
            name = str(raw_name).strip()
            if not name:
                continue
            old = existing.get(name)
            if old is None:
                profiles.append(ClassProfile(name, class_id))
            else:
                clone = ClassProfile.from_dict(old.to_dict(), default_id=class_id)
                clone.class_id = class_id
                profiles.append(clone)
        return ClassProfiles(profiles)

    def save(self, path: Path) -> None:
        save_class_profiles(path, self)

    @classmethod
    def load(cls, path: Path, classes: Iterable[str]) -> "ClassProfiles":
        return load_class_profiles(path, classes)


ClassProfileRegistry = ClassProfiles
ClassProfileManager = ClassProfiles


def default_profiles(classes: Iterable[str]) -> ClassProfiles:
    return ClassProfiles(ClassProfile(str(name).strip(), index) for index, name in enumerate(classes) if str(name).strip())


def load_class_profiles(path: Path, classes: Iterable[str]) -> ClassProfiles:
    """Load old/new JSON shapes and reconcile ids with the GUI class order."""

    raw = load_json(path, {})
    values: Any = raw.get("profiles", []) if isinstance(raw, Mapping) else raw
    if isinstance(values, Mapping):
        values = [dict(value, class_name=name) for name, value in values.items() if isinstance(value, Mapping)]
    profiles: list[ClassProfile] = []
    if isinstance(values, list):
        for index, item in enumerate(values):
            if isinstance(item, Mapping):
                try:
                    profile = ClassProfile.from_dict(item, default_id=index)
                except (TypeError, ValueError):
                    continue
                if profile.class_name:
                    profiles.append(profile)
    return ClassProfiles(profiles).sync_classes(classes)


def save_class_profiles(path: Path, profiles: ClassProfiles | Iterable[ClassProfile]) -> None:
    registry = profiles if isinstance(profiles, ClassProfiles) else ClassProfiles(profiles)
    atomic_write_json(path, registry.to_dict())


def ensure_class_profiles(path: Path, classes: Iterable[str]) -> ClassProfiles:
    classes = list(classes)
    registry = load_class_profiles(path, classes) if path.exists() else default_profiles(classes)
    # Always persist a valid file; this makes old projects self-upgrading while
    # keeping the GUI's class ordering authoritative.
    save_class_profiles(path, registry)
    return registry


load_or_create_class_profiles = ensure_class_profiles


__all__ = [
    "ClassProfile",
    "ClassProfiles",
    "ClassProfileRegistry",
    "ClassProfileManager",
    "default_profiles",
    "load_class_profiles",
    "save_class_profiles",
    "ensure_class_profiles",
    "load_or_create_class_profiles",
]
