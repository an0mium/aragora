#!/usr/bin/env python3
"""Seed the live dashboard with realistic demo data so visitors see a working
platform instead of empty panels.  Populates the actual SQLite databases that
dashboard API endpoints read from (ELO, debates, trending, tournaments).

Usage:
    python scripts/seed_demo.py              # Seed all data
    python scripts/seed_demo.py --clear      # Clear demo data first
    python scripts/seed_demo.py --check      # Just check if data exists
"""
from __future__ import annotations
import argparse, json, logging, random, sys, time, uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("seed_demo")
random.seed(42)

# -- Demo content -----------------------------------------------------------
DEBATES = [
    ("Should we adopt microservices or keep the monolith?", "architecture", True, 0.91),
    ("Is React or Vue better for enterprise dashboards?", "frontend", True, 0.84),
    ("Should AI-generated code require human review?", "engineering", True, 0.96),
    ("Build vs buy for our observability stack?", "infrastructure", False, 0.52),
    ("Should we migrate from REST to GraphQL?", "api_design", True, 0.79),
    ("Is Rust worth the learning curve for our backend?", "engineering", True, 0.73),
    ("Should we open-source our SDK?", "strategy", True, 0.88),
    ("Kubernetes vs serverless for our next deployment?", "infrastructure", True, 0.81),
    ("Should we enforce 90% code coverage?", "quality", False, 0.48),
    ("Is it time to drop Python 3.9 support?", "compatibility", True, 0.93),
]
AGENTS = [  # (name, elo, wins, losses, draws, debates)
    ("claude-opus", 1782, 47, 12, 6, 65), ("gpt-4o", 1721, 41, 17, 7, 65),
    ("gemini-pro", 1654, 35, 22, 8, 65),  ("mistral-large", 1598, 30, 25, 10, 65),
    ("grok-2", 1543, 27, 28, 10, 65),     ("deepseek-v3", 1487, 23, 31, 11, 65),
    ("llama-405b", 1412, 18, 35, 12, 65), ("qwen-72b", 1356, 14, 39, 12, 65),
]
TRENDING = [
    ("AI agents are replacing junior developers", "hackernews", "ai", 342),
    ("New NIST post-quantum cryptography standard released", "arxiv", "security", 189),
    ("Rust memory safety approach adopted by Linux 6.14", "github", "systems", 567),
    ("LLM context windows now exceed 10M tokens", "arxiv", "ai", 231),
    ("Remote work productivity data after 5 years", "hackernews", "culture", 418),
]
RISKS = [
    ("high", "3 agents show calibration drift >15% in security domain"),
    ("medium", "Consensus confidence below SLO target for architecture debates"),
    ("low", "ELO variance increasing for deepseek-v3 over last 20 matches"),
]
TOURN_AGENTS = ["claude-opus", "gpt-4o", "gemini-pro", "mistral-large"]
_DEMO_LIKE = "demo_%"

def _data_dir() -> Path:
    try:
        from aragora.persistence.db_config import get_default_data_dir
        d = get_default_data_dir()
    except ImportError:
        d = Path(".nomic")
    d.mkdir(parents=True, exist_ok=True); return d

def _past(days=0, hours=0):
    return (datetime.now(timezone.utc) - timedelta(days=days, hours=hours)).isoformat()


# -- Seed ELO ---------------------------------------------------------------
def seed_elo(clear: bool) -> int:
    try:
        from aragora.ranking.elo import EloSystem, AgentRating
    except ImportError:
        logger.warning("EloSystem not importable, skipping"); return 0
    elo = EloSystem()
    if clear:
        with elo._db.connection() as c:
            for tbl in ("ratings", "matches", "elo_history"):
                try:
                    c.execute(f"DELETE FROM {tbl}")
                except Exception:
                    pass
        logger.info("  Cleared ELO data")
    count = 0
    for name, rating, wins, losses, draws, debates in AGENTS:
        if elo.get_rating(name, use_cache=False).games_played > 0 and not clear:
            continue
        ar = AgentRating(
            agent_name=name, elo=float(rating),
            domain_elos={"engineering": rating + random.randint(-80, 80),
                         "architecture": rating + random.randint(-80, 80)},
            wins=wins, losses=losses, draws=draws, debates_count=debates,
            critiques_accepted=random.randint(20, 60),
            critiques_total=random.randint(60, 100),
            calibration_correct=random.randint(10, 30),
            calibration_total=random.randint(30, 50),
            calibration_brier_sum=random.uniform(3.0, 8.0),
            updated_at=datetime.now(timezone.utc).isoformat(),
        )
        elo._save_rating(ar)
        elo._record_elo_history(name, float(rating), debate_id="demo_seed")
        count += 1
    # 20 match history rows so recent-matches API is populated
    names = [a[0] for a in AGENTS]
    for i in range(20):
        a1, a2 = random.sample(names, 2)
        w = random.choice([a1, a2, None])
        s = {a1: 1.0 if w == a1 else (0.5 if w is None else 0.0),
             a2: 1.0 if w == a2 else (0.5 if w is None else 0.0)}
        try:
            elo._save_match(f"demo_match_{i:03d}", w, [a1, a2],
                            random.choice(["engineering", "architecture", "security"]),
                            s, {a1: random.uniform(-15, 15), a2: random.uniform(-15, 15)})
        except Exception:
            pass
    return count

# -- Seed debates (DebateStorage) -------------------------------------------
def seed_debates(clear: bool) -> int:
    try:
        from aragora.server.storage import DebateStorage
    except ImportError:
        logger.warning("DebateStorage not importable, skipping"); return 0
    st = DebateStorage(); pool = [a[0] for a in AGENTS]; count = 0
    if clear:
        with st.connection() as c:
            c.execute("DELETE FROM debates WHERE id LIKE ?", (_DEMO_LIKE,))
        logger.info("  Cleared debate data")
    for idx, (task, domain, consensus, conf) in enumerate(DEBATES):
        did = f"demo_debate_{idx:03d}"
        with st.connection() as c:
            if c.execute("SELECT 1 FROM debates WHERE id=?", (did,)).fetchone() and not clear:
                continue
        agents = random.sample(pool, random.randint(3, 5))
        created = datetime.now(timezone.utc) - timedelta(days=random.randint(1, 28),
                                                          hours=random.randint(0, 23))
        artifact = json.dumps({
            "artifact_id": did, "task": task, "agents": agents, "rounds": 3,
            "messages": [{"agent": ag, "round": r,
                          "content": f"Round {r} analysis of: {task}",
                          "timestamp": (created + timedelta(seconds=r * 60)).isoformat()}
                         for r in range(1, 4) for ag in agents],
            "consensus_proof": {"reached": consensus, "confidence": conf,
                                "method": "supermajority"},
            "domain": domain, "duration_seconds": random.randint(45, 300),
            "created_at": created.isoformat(), "metadata": {"demo": True}})
        slug = st.generate_slug(task)
        with st.connection() as c:
            c.execute("""INSERT OR REPLACE INTO debates
                         (id,slug,task,agents,artifact_json,consensus_reached,confidence,created_at)
                         VALUES (?,?,?,?,?,?,?,?)""",
                      (did, slug, task, json.dumps(agents), artifact,
                       consensus, conf, created.isoformat()))
        count += 1
    return count

# -- Seed CritiqueStore debates ---------------------------------------------
def seed_critique_debates(clear: bool) -> int:
    try:
        from aragora.memory.store import CritiqueStore
    except ImportError:
        logger.warning("CritiqueStore not importable, skipping"); return 0
    st = CritiqueStore(); count = 0
    if clear:
        with st.connection() as c:
            c.execute("DELETE FROM debates WHERE id LIKE ?", (_DEMO_LIKE,))
        logger.info("  Cleared critique data")
    for idx, (task, _dom, consensus, conf) in enumerate(DEBATES):
        did = f"demo_debate_{idx:03d}"
        with st.connection() as c:
            if c.execute("SELECT 1 FROM debates WHERE id=?", (did,)).fetchone() and not clear:
                continue
        verdict = f"Consensus conclusion for: {task}" if consensus else None
        with st.connection() as c:
            c.execute("""INSERT INTO debates
                         (id,task,final_answer,consensus_reached,confidence,rounds_used,
                          duration_seconds,created_at)
                         VALUES (?,?,?,?,?,?,?,?)""",
                      (did, task, verdict, int(consensus), conf, 3,
                       random.uniform(45, 300), _past(days=random.randint(1, 28))))
        count += 1
    return count

# -- Seed trending topics + risk warnings -----------------------------------
def seed_trending(clear: bool) -> int:
    try:
        from aragora.pulse.store import ScheduledDebateStore
    except ImportError:
        logger.warning("ScheduledDebateStore not importable, skipping"); return 0
    st = ScheduledDebateStore(); count = 0
    if clear:
        with st.connection() as c:
            c.execute("DELETE FROM scheduled_debates WHERE id LIKE ?", (_DEMO_LIKE,))
        logger.info("  Cleared trending data")

    def _insert(rec_id, topic, platform, category, volume, debate_id, consensus, conf, rounds):
        with st.connection() as c:
            if c.execute("SELECT 1 FROM scheduled_debates WHERE id=?",
                         (rec_id,)).fetchone() and not clear:
                return False
        h = ScheduledDebateStore.hash_topic(topic)
        with st.connection() as c:
            c.execute("""INSERT OR REPLACE INTO scheduled_debates
                         (id,topic_hash,topic_text,platform,category,volume,debate_id,
                          created_at,consensus_reached,confidence,rounds_used,scheduler_run_id)
                         VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                      (rec_id, h, topic, platform, category, volume, debate_id,
                       time.time() - random.randint(0, 86400), consensus, conf,
                       rounds, "demo_run_001"))
        return True

    for i, (topic, plat, cat, vol) in enumerate(TRENDING):
        linked = i < 3
        count += _insert(f"demo_trend_{i:03d}", topic, plat, cat, vol,
                         f"demo_debate_{i:03d}" if linked else None,
                         1 if linked else None,
                         round(random.uniform(0.7, 0.95), 2) if linked else None,
                         3 if linked else 0)
    for i, (sev, desc) in enumerate(RISKS):
        count += _insert(f"demo_risk_{i:03d}", desc, "internal", f"risk_{sev}",
                         random.randint(1, 100), None, None, None, 0)
    return count

# -- Seed tournament --------------------------------------------------------
def seed_tournament(clear: bool) -> int:
    try:
        from aragora.ranking.tournaments import TournamentManager
    except ImportError:
        logger.warning("TournamentManager not importable, skipping"); return 0
    mgr = TournamentManager()
    if clear:
        with mgr._get_connection() as c:
            c.execute("DELETE FROM tournaments WHERE name LIKE 'Demo%'")
            c.execute("DELETE FROM matches")
        logger.info("  Cleared tournament data")
    for t in mgr.list_tournaments(limit=10):
        if t.name.startswith("Demo"):
            logger.info("  Tournament exists, skipping"); return 0
    tourn = mgr.create_tournament("Demo Weekly Championship", TOURN_AGENTS, "round_robin")
    for m in mgr.get_matches(tournament_id=tourn.tournament_id):
        w = random.choice([m.agent1, m.agent2, None])
        s1, s2 = round(random.uniform(0.3, 1.0), 2), round(random.uniform(0.3, 1.0), 2)
        if w == m.agent1: s1 = max(s1, s2 + 0.1)
        elif w == m.agent2: s2 = max(s1 + 0.1, s2)
        mgr.record_match_result(m.match_id, w, s1, s2, update_elo=False)
    return len(mgr.get_matches(tournament_id=tourn.tournament_id))

# -- Check ------------------------------------------------------------------
def _safe_count(fn):
    try: return fn()
    except Exception: return 0

def check_data() -> dict[str, int]:
    def _agents():
        from aragora.ranking.elo import EloSystem
        return len(EloSystem().list_agents())
    def _debates():
        from aragora.server.storage import DebateStorage
        with DebateStorage().connection() as c:
            return c.execute("SELECT COUNT(*) FROM debates WHERE id LIKE ?",
                             (_DEMO_LIKE,)).fetchone()[0]
    def _trending():
        from aragora.pulse.store import ScheduledDebateStore
        with ScheduledDebateStore().connection() as c:
            return c.execute("SELECT COUNT(*) FROM scheduled_debates WHERE id LIKE ?",
                             (_DEMO_LIKE,)).fetchone()[0]
    def _tourn():
        from aragora.ranking.tournaments import TournamentManager
        return len([t for t in TournamentManager().list_tournaments(50)
                    if t.name.startswith("Demo")])
    return {"agents": _safe_count(_agents), "debates": _safe_count(_debates),
            "trending": _safe_count(_trending), "tournaments": _safe_count(_tourn)}

# -- Main -------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Seed Aragora dashboard with demo data")
    ap.add_argument("--clear", action="store_true", help="Clear demo data first")
    ap.add_argument("--check", action="store_true", help="Just check if data exists")
    args = ap.parse_args()
    dd = _data_dir(); logger.info(f"Data directory: {dd}")
    if args.check:
        c = check_data()
        print("\nExisting demo data:")
        for k, v in c.items(): print(f"  {k:15s}: {v}")
        print(f"  {'total':15s}: {sum(c.values())}")
        return 0 if sum(c.values()) > 0 else 1
    steps = [("agents", seed_elo), ("debates", seed_debates),
             ("critique_debates", seed_critique_debates),
             ("trending_and_risks", seed_trending),
             ("tournament_matches", seed_tournament)]
    r = {}
    for name, fn in steps:
        logger.info(f"Seeding {name}...")
        r[name] = fn(args.clear)
    print(f"\n{'='*50}\nDEMO DATA SEEDED\n{'='*50}")
    for k, v in r.items():
        print(f"  {k:25s}: {v} created" if v else f"  {k:25s}: skipped (exists)")
    print(f"{'='*50}\n  Data directory: {dd}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
