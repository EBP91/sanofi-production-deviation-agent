import os
import random
from datetime import datetime, timedelta

# Konfiguration
OUTPUT_DIR = "data"
NUM_DEVIATIONS = 40   # Genug für eine Suche, nicht zu viele für Übersicht
NUM_SOPS = 4          # Eine perfekte SOP pro Abteilung

# --- DEFINITION DER SZENARIEN ---
# Wir nutzen einfache, aber realistische Pharma-Probleme.
# Jedes Szenario enthält: Problembeschreibung, Wahre Ursache, Lösung.

SCENARIOS = {
    "UPSTREAM": {
        "dept_name": "Upstream_Fermentation",
        "equipment": ["Bioreaktor-100L", "Bioreaktor-500L"],
        "process": "Zellkultur-Wachstum",
        "cases": [
            {
                "issue": "pH-Wert sinkt zu schnell ab",
                "root_cause": "Die pH-Sonde war verschmutzt (Biofilm) und hat falsche Werte geliefert.",
                "action": "Sonde gereinigt und neu kalibriert.",
                "capa": "Reinigungsintervall der Sonden verkürzt."
            },
            {
                "issue": "Temperatur zu hoch (über 37.5 Grad)",
                "root_cause": "Das Kühlwasserventil hat geklemmt und nicht geöffnet.",
                "action": "Ventil manuell geöffnet, Techniker gerufen.",
                "capa": "Ventil gegen ein digitales Regelventil ausgetauscht."
            }
        ],
        "sop_steps": """1. Vorbereitung
- Stellen Sie sicher, dass der Bioreaktor sauber und steril ist.
- Kalibrieren Sie die pH-Sonde (Muss-Wert: 7.0) und den Sauerstoffsensor.

2. Prozessstart
- Füllen Sie das Nährmedium ein.
- Starten Sie das Rührwerk mit 100 Umdrehungen pro Minute (RPM).
- Die Temperatur muss konstant auf 37.0 Grad Celsius gehalten werden (+/- 0.5 Grad).

3. Überwachung
- Prüfen Sie stündlich den pH-Wert. Sollbereich: 6.8 bis 7.2.
- Prüfen Sie, ob genug Kühlwasser verfügbar ist."""
    },
    
    "DOWNSTREAM": {
        "dept_name": "Downstream_Reinigung",
        "equipment": ["Chromatographie-Saeule-A", "Filter-Station-B"],
        "process": "Protein-Reinigung",
        "cases": [
            {
                "issue": "Filter verstopft (Druck zu hoch)",
                "root_cause": "Die Lösung war zu kalt, dadurch wurde das Produkt zähflüssig.",
                "action": "Prozess pausiert, Lösung auf Raumtemperatur erwärmt.",
                "capa": "Temperaturfühler am Vorratstank installiert."
            },
            {
                "issue": "Undichtigkeit am Schlauch",
                "root_cause": "Eine Dichtung (O-Ring) wurde beim Zusammenbau vergessen.",
                "action": "Prozess gestoppt, Dichtung eingesetzt.",
                "capa": "Vier-Augen-Prinzip beim Zusammenbau eingeführt."
            }
        ],
        "sop_steps": """1. Aufbau
- Kontrollieren Sie alle Schläuche auf Risse.
- Setzen Sie überall neue Dichtungen ein.
- Ziehen Sie die Klemmen fest an.

2. Durchführung
- Starten Sie die Pumpe langsam (Flussrate: 5 Liter/Stunde).
- Der Druck darf 3.0 bar nicht überschreiten.
- Wenn der Druck über 3.0 bar steigt: Sofort stoppen!

3. Filterwechsel
- Filter müssen nach jeder Charge gewechselt werden."""
    },

    "FILLING": {
        "dept_name": "Abfuellung_Steril",
        "equipment": ["Fuelllinie-Isolator", "Capping-Maschine"],
        "process": "Abfüllung",
        "cases": [
            {
                "issue": "Glassplitter im Isolator gefunden",
                "root_cause": "Ein Fläschchen (Vial) ist am Sternrad zerbrochen, weil es falsch justiert war.",
                "action": "Linie gestoppt, alles gereinigt (Line Clearance).",
                "capa": "Justierung des Sternrads durch Mechaniker überprüft."
            },
            {
                "issue": "Füllgewicht zu niedrig",
                "root_cause": "Luftblasen im Zuleitungsschlauch.",
                "action": "Leitung entlüftet und 10 Test-Füllungen gemacht.",
                "capa": "Automatisches Entlüftungsventil eingebaut."
            }
        ],
        "sop_steps": """1. Sterilität
- Der Isolator darf nur über die Handschuhe bedient werden.
- Prüfen Sie die Handschuhe vor jeder Schicht auf Risse.

2. Abfüllen
- Zielgewicht pro Fläschchen: 10.0 Gramm (+/- 0.2g).
- Führen Sie alle 30 Minuten eine Gewichtskontrolle durch.
- Bei Glasbruch: Anlage sofort stoppen und Reinigunsprotokoll starten.

3. Abschluss
- Verschließen Sie die Fläschchen sofort mit Gummistopfen."""
    },

    "PACKAGING": {
        "dept_name": "Verpackung",
        "equipment": ["Kartonierer-HighSpeed", "Etikettierer"],
        "process": "Verpackung",
        "cases": [
            {
                "issue": "Etikett schief aufgeklebt",
                "root_cause": "Der Sensor für die Flaschenerkennung war verstaubt.",
                "action": "Sensor gereinigt.",
                "capa": "Reinigungsplan für Sensoren erstellt (täglich)."
            },
            {
                "issue": "Beipackzettel fehlt",
                "root_cause": "Magazin für Beipackzettel war leer und Alarm ging nicht.",
                "action": "Magazin aufgefüllt, betroffene Packungen aussortiert.",
                "capa": "Warnleuchte repariert."
            }
        ],
        "sop_steps": """1. Materialprüfung
- Prüfen Sie, ob die richtigen Etiketten und Beipackzettel eingelegt sind.
- Vergleichen Sie die Materialnummer mit dem Auftrag.

2. Maschinenstart
- Starten Sie den Kartonierer.
- Achten Sie darauf, dass immer genug Beipackzettel im Magazin sind.

3. Qualitätskontrolle
- Die Kamera prüft, ob das Etikett gerade sitzt.
- Wenn die Kamera 3 Fehler hintereinander meldet: Maschine anhalten und Sensor prüfen."""
    }
}

# --- TEMPLATES ---

SOP_TEMPLATE = """# SOP: {process}
**Dokument-ID:** SOP-{dept_short}-{id:02d}  
**Abteilung:** {dept_name}  
**Anlage:** {equipment}  
**Gültig ab:** 01.01.2024

## 1. Ziel
Diese Arbeitsanweisung beschreibt den korrekten Ablauf für den Prozess "{process}". 
Sicherheit und Qualität stehen an erster Stelle.

## 2. Arbeitsanweisungen (Schritt-für-Schritt)
{steps}

## 3. Fehlerbehebung (Troubleshooting)
Wenn Probleme auftreten, nutzen Sie diese Tabelle:

| Problem | Mögliche Ursache | Sofortmaßnahme |
| :--- | :--- | :--- |
| {issue_1} | {cause_1} | {action_1} |
| {issue_2} | {cause_2} | {action_2} |

## 4. Wichtige Grenzwerte
- Halten Sie sich strikt an die oben genannten Zahlenwerte (Temperatur, Druck, Gewicht).
- Dokumentieren Sie jede Abweichung sofort.
"""

DEVIATION_TEMPLATE = """# ABWEICHUNGSBERICHT (DEVIATION)
**ID:** DEV-{year}-{id:04d}  
**Datum:** {date}  
**Anlage:** {equipment}  
**Abteilung:** {dept_name}

## Beschreibung des Fehlers
Während des Prozesses "{process}" ist folgendes passiert:  
**{issue}** Der Fehler trat um {time} Uhr auf.

## Ursachenanalyse (Root Cause)
Wir haben untersucht, warum das passiert ist:  
{root_cause}

## Durchgeführte Maßnahmen
Um das Problem sofort zu lösen, haben wir folgendes getan:  
{action}

## Langfristige Lösung (CAPA)
Damit das nicht wieder passiert:  
{capa}

**Status:** Geschlossen.  
**Unterschrift:** {operator}
"""

OPERATORS = ["M. Müller", "S. Klein", "A. Yilmaz", "J. Schmidt"]

def generate_files():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    print(f"Generiere daten in '{OUTPUT_DIR}'...")

    # 1. SOPs generieren (4 Stück, eine pro Abteilung)
    sop_count = 0
    for key, data in SCENARIOS.items():
        sop_count += 1
        # Wir nehmen die beiden definierten Fälle für die Troubleshooting-Tabelle
        case1 = data["cases"][0]
        case2 = data["cases"][1]
        
        content = SOP_TEMPLATE.format(
            process=data["process"],
            dept_short=data["dept_name"][:3].upper(),
            id=sop_count,
            dept_name=data["dept_name"],
            equipment="Alle Typen", # SOP gilt generell
            steps=data["sop_steps"],
            # Fülle die Tabelle in der SOP automatisch mit den bekannten Problemen
            issue_1=case1["issue"], cause_1=case1["root_cause"], action_1=case1["action"],
            issue_2=case2["issue"], cause_2=case2["root_cause"], action_2=case2["action"]
        )
        
        filename = f"{OUTPUT_DIR}/SOP_{data['dept_name']}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)

    # 2. Deviations generieren (viele, basierend auf den Cases)
    for i in range(1, NUM_DEVIATIONS + 1):
        # Zufällige Abteilung wählen
        key = random.choice(list(SCENARIOS.keys()))
        data = SCENARIOS[key]
        
        # Zufälligen Fall aus dieser Abteilung wählen
        case = random.choice(data["cases"])
        
        # Datum generieren
        random_days = random.randint(0, 365)
        date_obj = datetime.now() - timedelta(days=random_days)
        date_str = date_obj.strftime("%d.%m.%Y")
        year_str = date_obj.strftime("%Y")

        content = DEVIATION_TEMPLATE.format(
            year=year_str,
            id=i,
            date=date_str,
            equipment=random.choice(data["equipment"]),
            dept_name=data["dept_name"],
            process=data["process"],
            issue=case["issue"],
            root_cause=case["root_cause"],
            action=case["action"],
            capa=case["capa"],
            time=f"{random.randint(6, 20)}:{random.randint(10, 59):02d}",
            operator=random.choice(OPERATORS)
        )
        
        filename = f"{OUTPUT_DIR}/DEV_{year_str}_{i:03d}_{data['dept_name'][:3]}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)

    print(f"Fertig! {sop_count} SOPs und {NUM_DEVIATIONS} Fehlerberichte erstellt.")

if __name__ == "__main__":
    generate_files()