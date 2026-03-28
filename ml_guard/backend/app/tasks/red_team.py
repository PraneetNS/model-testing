from app.core.celery_app import celery_app
from app.db.session import SessionLocal
from app.db.models import RedTeamSession, RedTeamAttack, Model
from app.services.red_team.agent import AdversarialAgent
from app.core.security import encrypt_content
import structlog
import time

logger = structlog.get_logger()

@celery_app.task(name="app.tasks.red_team.execute_red_team_campaign")
def execute_red_team_campaign(session_id: str, max_attacks: int = 10):
    """
    Background worker process for adversarial testing.
    Executes an iterative, multi-round attack loop against target LLMs.
    """
    db = SessionLocal()
    try:
        session = db.query(RedTeamSession).get(session_id)
        if not session:
            logger.error("RedTeamSession not found", session_id=session_id)
            return
            
        model = db.query(Model).get(session.model_id)
        agent = AdversarialAgent(target_url=model.artifact_url) # Assume URL is in artifact_url
        
        categories = ["jailbreak", "pii", "bias", "injection", "role_confusion"]
        
        for i in range(max_attacks):
            category = categories[i % len(categories)]
            logger.info("Starting attack round", round=i+1, category=category)
            
            # 1. Generate & Refine (Multi-round loop within agent)
            attack_result = agent.run_attack_sequence(category)
            
            # 2. Persist findings
            attack = RedTeamAttack(
                session_id=session.id,
                category=category,
                severity=attack_result["severity"],
                rounds=attack_result["rounds"],
                is_successful=attack_result["is_successful"],
                encrypted_prompt=encrypt_content(attack_result["prompt"]),
                encrypted_response=encrypt_content(attack_result["response"]) if attack_result["response"] else None,
                judge_reasoning=attack_result["reasoning"]
            )
            db.add(attack)
            
            # 3. Update session
            session.total_attacks += 1
            if attack_result["is_successful"]:
                session.success_count += 1
            
            db.commit()
            
            # Rate limiting / Backpressure
            time.sleep(6) # Max 10 per min
            
        session.status = "COMPLETED"
        db.commit()
        logger.info("Campaign completed", session_id=session_id, success_rate=session.success_count/session.total_attacks)
        
    except Exception as e:
        logger.error("Red Team campaign failed", error=str(e), session_id=session_id)
        if session:
            session.status = "FAILED"
            db.commit()
    finally:
        db.close()
