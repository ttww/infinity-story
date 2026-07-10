"""Predefined story fragment templates for enriching nodes.

Templates are short, genre-agnostic narrative building blocks that can be
inserted into a node's scene_text or scene_goal to quickly flesh out
a story without writing everything from scratch.

Each template returns a dict with keys matching StoryNode fields:
    scene_text, scene_goal, mood, reveals, quality_notes, etc.
"""

from __future__ import annotations

from typing import Any


_TEMPLATES: dict[str, dict[str, Any]] = {
    # ── Atmospheric / mood-setting ──────────────────────────────────
    "arrival_at_location": {
        "name": "Ankunft am Ort",
        "description": "Eine atmosphärische Ankunftsszene mit detaillierter Ortsbeschreibung.",
        "category": "atmosphere",
        "fields": {
            "scene_text": (
                "Die Luft war schwer von feuchtem Nebel, der sich über den nassen Boden zog. "
                "Jeder Schritt hallte auf dem kalten Stein, als wäre der Ort selbst ein lebendes Wesen, "
                "das den Ankömmling prüfte. In der Ferne flackerte ein Licht — unsicher, ob es Hoffnung "
                "oder eine Falle bedeutete."
            ),
            "mood": "atmosphärisch_unheimlich",
            "quality_notes": ["Ortsbeschreibung etabliert", "Sinnliche Details: Geruch, Klang, Licht"],
        },
    },
    "tension_rising": {
        "name": "Steigende Spannung",
        "description": "Spannungsaufbau durch innere Unruhe und äußere Zeichen.",
        "category": "atmosphere",
        "fields": {
            "scene_text": (
                "Etwas stimmte nicht. Die Stille war zu tief, zu vollständig — als hätte die Welt "
                "den Atem angehalten. Ein leises Knirschen hinter der Wand. Dann noch eines. "
                "Das Herz schlug schneller, und die Hände begannen zu zittern."
            ),
            "mood": "spannung_steigend",
        },
    },
    "calm_before_storm": {
        "name": "Stille vor dem Sturm",
        "description": "Ein ruhiger Moment, der eine kommende Krise ankündigt.",
        "category": "atmosphere",
        "fields": {
            "scene_text": (
                "Es war der Art von Frieden, die man nur spürt, wenn etwas Schlimmes naht. "
                "Die Sonne sank blutrot hinter den Dächern, und für einen kurzen Moment "
                "schienen alle Sorgen klein. Dann fiel die Nacht."
            ),
            "mood": "ruhig_vor_sturm",
        },
    },

    # ── Character interactions ─────────────────────────────────────
    "first_encounter": {
        "name": "Erste Begegnung",
        "description": "Erstkontakt mit einer neuen Figur — Misstrauen und Neugier.",
        "category": "character",
        "fields": {
            "scene_text": (
                "Die Tür öffnete sich knarrend, und dahinter stand eine Gestalt, "
                "die weder Freund noch Feind zu sein schien. Die Augen waren wachsam, "
                "die Hände bereit, aber nicht bedrohlich. "
                "\n\n„Du bist also derjenige, den sie schicken\", sagte die Stimme."
            ),
            "mood": "misstrauisch_neugierig",
            "quality_notes": ["Neue Figur eingeführt", "Spannung zwischen Misstrauen und Neugier"],
        },
    },
    "betrayal_reveal": {
        "name": "Verrat aufgedeckt",
        "description": "Ein Verbündeter entpuppt sich als Verräter.",
        "category": "character",
        "fields": {
            "scene_text": (
                "„Es tut mir leid.“ Die Worte kamen so leise, dass sie fast im Wind untergingen. "
                "Aber das Messer in der Hand sprach eine lautere Sprache. "
                "\n\nAll die Monate. All die gemeinsamen Nächte am Feuer. "
                "Es bedeutete nichts. Es hatte nie etwas bedeutet."
            ),
            "mood": "schock_verrat",
            "reveals": ["Der Verbündete war ein Verräter"],
        },
    },
    "emotional_bond": {
        "name": "Emotionales Band",
        "description": "Ein Moment der Verbundenheit zwischen Figuren.",
        "category": "character",
        "fields": {
            "scene_text": (
                "Sie saßen nebeneinander auf der alten Mauer und sahen den Sternen zu. "
                "Keiner sprach, aber es gab nichts zu sagen, was die Stille nicht schon sagte. "
                "Zum ersten Mal seit Wochen fühlte sich die Welt nicht wie ein Feind an."
            ),
            "mood": "warm_vertraut",
        },
    },

    # ── Plot / mystery ─────────────────────────────────────────────
    "clue_discovery": {
        "name": "Hinweis entdeckt",
        "description": "Ein wichtiger Hinweis wird gefunden — mit Fragen, nicht Antworten.",
        "category": "plot",
        "fields": {
            "scene_text": (
                "Unter dem losen Stein lag ein zusammengefaltetes Papier, vergilbt und brüchig. "
                "Die Schrift war kaum lesbar, aber drei Worte standen deutlich: "
                "„Siehe nicht hin.“ Was sollte man nicht sehen? Und wer hatte dies hier versteckt?"
            ),
            "mood": "mysteriös",
            "reveals": ["Ein versteckter Hinweis wurde gefunden"],
            "quality_notes": ["Hinweis wirft neue Fragen auf", "Mysterium vertieft"],
        },
    },
    "confrontation": {
        "name": "Konfrontation",
        "description": "Direkte Auseinandersetzung mit dem Antagonisten.",
        "category": "plot",
        "fields": {
            "scene_text": (
                "„Du denkst, du hast gewonnen.“ Die Stimme kam von überall und nirgends. "
                "„Aber du kennst nicht einmal die Regeln des Spiels, das du spielst.\" "
                "\n\nDie Gestalt trat aus dem Schatten, und zum ersten Mal sah man das Gesicht "
                "des Feindes. Es war nicht das eines Monsters. Es war das eines Mannes, "
                "der zu lange allein war."
            ),
            "mood": "konfrontativ",
            "quality_notes": ["Antagonist erhält Gesichter/Menschlichkeit", "Höhepunkt vorbereitet"],
        },
    },
    "moral_dilemma": {
        "name": "Moralisches Dilemma",
        "description": "Eine Wahl zwischen zwei gleichermaßen schwierigen Wegen.",
        "category": "plot",
        "fields": {
            "scene_text": (
                "Zwei Wege lagen vor ihr. Der eine führte zur Wahrheit — aber er würde jemanden "
                "verraten, der ihr vertraute. Der andere führte zur Sicherheit — aber er bedeutete, "
                "die Wahrheit für immer zu begraben. Konnte sie mit der einen leben? Konnte sie "
                "mit der anderen sterben?"
            ),
            "mood": "zwiespältig",
            "quality_notes": ["Keine einfache Lösung", "Beide Optionen haben Konsequenzen"],
        },
    },

    # ── Action / combat ────────────────────────────────────────────
    "chase_sequence": {
        "name": "Verfolgungsjagd",
        "description": "Schnelle, temporeiche Verfolgungsszene.",
        "category": "action",
        "fields": {
            "scene_text": (
                "Schritte hinter ihr, schnell, zu schnell. Sie bog scharf um die Ecke, "
                "riss eine Mülltonne um, hörte sie krachend zu Boden gehen. "
                "Die Verfolger zögerten — einen Moment. Dann kamen sie wieder, näher. "
                "Lungen brannten, Beine schrien. Nur noch zwei Straßenecken. Nur noch eine."
            ),
            "mood": "hetzt_temporeich",
        },
    },
    "combat_climax": {
        "name": "Kampf-Höhepunkt",
        "description": "Entscheidender Kampfmoment mit hohem Einsatz.",
        "category": "action",
        "fields": {
            "scene_text": (
                "Stahl klirrte auf Stahl. Der Schlag ging durch die Arme bis in die Schultern. "
                "Kein Zurück mehr. Der nächste Angriff kam von links — zu spät gesehen, "
                "nur noch eine Parade, die kaum hielt. Dann eine Lücke. Eine einzige Lücke. "
                "Und in der Lücke: eine Entscheidung."
            ),
            "mood": "entscheidend_intensiv",
        },
    },

    # ── Ending / resolution ────────────────────────────────────────
    "bittersweet_ending": {
        "name": "Bittersüßes Ende",
        "description": "Ein Sieg mit Preis — melancholisch, aber befriedigend.",
        "category": "ending",
        "fields": {
            "scene_text": (
                "Es war vorbei. Wirklich vorbei. Die Sonne ging auf über einer Welt, "
                "die gerettet war — oder zumindest eine Chance hatte. "
                "Aber die Stille neben ihr, wo einst jemand gestanden hatte, "
                "war ein Preis, den keine Siegesfeier je füllen konnte."
            ),
            "mood": "bittersüß",
            "quality_notes": ["Auflösung mit emotionalem Preis", "Befriedigend aber nicht einfach"],
        },
    },
    "twist_ending": {
        "name": "Twist-Ende",
        "description": "Eine unerwartete Wendung, die alles neu kontextualisiert.",
        "category": "ending",
        "fields": {
            "scene_text": (
                "„Glückwunsch.\" Die Stimme kam von hinten. Vertraut. Zu vertraut. "
                "Sie drehte sich um. Und sah sich selbst. "
                "\n\n„Du hast alles richtig gemacht\", sagte der Spiegel. "
                "„Du musstest es nur von der anderen Seite sehen.\""
            ),
            "mood": "schockierend_twist",
            "reveals": ["Die ganze Geschichte wurde aus der falschen Perspektive erzählt"],
            "quality_notes": ["Twist wirft gesamte Geschichte neu auf", "Kontinuität muss stimmen"],
        },
    },

    # ── Transition / pacing ────────────────────────────────────────
    "time_passage": {
        "name": "Zeitsprung",
        "description": "Ein Montage-artiger Übergang mit Zeitverlauf.",
        "category": "transition",
        "fields": {
            "scene_text": (
                "Die Tage verschwammen zu Wochen, die Wochen zu Monaten. "
                "Die Wunden heilten — die meisten. Die Narben blieben, "
                "sichtbare und unsichtbare. Die Welt drehte sich weiter, "
                "obwohl sich manche wünschten, sie würde stillstehen."
            ),
            "mood": "melancholisch_vergangen",
        },
    },
    "scene_transition": {
        "name": "Szenenübergang",
        "description": "Glatter Übergang zwischen zwei Szenen/Orten.",
        "category": "transition",
        "fields": {
            "scene_text": (
                "Als sie die Tür hinter sich schloss, war die Welt draußen eine andere geworden. "
                "Der Nebel hatte sich verzogen, aber etwas Neues war in der Luft — "
                "etwas, das nach Veränderung roch."
            ),
            "mood": "übergang_hoffnungsvoll",
        },
    },
}


def list_templates() -> list[dict[str, Any]]:
    """Return all available templates as a list of summary dicts."""
    result = []
    for key, tpl in _TEMPLATES.items():
        result.append({
            "id": key,
            "name": tpl["name"],
            "description": tpl["description"],
            "category": tpl["category"],
        })
    return result


def get_template(template_id: str) -> dict[str, Any] | None:
    """Return a single template by ID, or None if not found."""
    tpl = _TEMPLATES.get(template_id)
    if tpl is None:
        return None
    return {
        "id": template_id,
        **tpl,
    }


def apply_template_to_node(template_id: str) -> dict[str, Any] | None:
    """Return the template fields to merge into a node.

    Returns None if the template doesn't exist.
    The caller is responsible for merging these fields into the node dict.
    """
    tpl = _TEMPLATES.get(template_id)
    if tpl is None:
        return None
    # Return a copy of the fields dict so the caller can't mutate the template
    return dict(tpl.get("fields", {}))


def list_categories() -> list[str]:
    """Return all unique category names."""
    return sorted({tpl["category"] for tpl in _TEMPLATES.values()})
