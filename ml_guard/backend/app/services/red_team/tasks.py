from app.core.celery_app import celery_app
from app.db.session import SessionLocal
from app.db.models import RedTeamSession, RedTeamAttack, Model
from app.services.red_team.agent import AdversarialAgent
from app.core.security import encrypt_content
from datetime import datetime
import asyncio
import structlog

logger = structlog.get_logger()

@celery_app.task(name="app.services.red_team.execute_red_team_campaign")
def execute_red_team_campaign(session_id: str, max_attacks: int = 10):
    """
    Background worker to execute the red-teaming autopilot.
    Iterates through categories, runs multi-round attacks, and persists results.
    """
    db = SessionLocal()
    try:
        session = db.query(RedTeamSession).get(session_id)
        if not session:
            logger.error("RedTeamSession not found", session_id=session_id)
            return

        model = db.query(Model).get(session.model_id)
        # Using the model's inference URL as the red-team target
        target_url = model.inference_url or "http://internal-mock-llm/v1/generate"
        agent = AdversarialAgent(target_url=target_url)

        # Attack Categories
        categories = ["jailbreak", "prompt_injection", "pii_extraction", "bias_elicitation"]
        
        attack_count = 0
        success_count = 0
        
        loop = asyncio.get_event_loop()
        
        for category in categories:
            if attack_count >= max_attacks:
                break
            
            # Multi-Round Escalation (Max 5 rounds)
            previous_refusal = None
            for r in range(1, 6):
                if attack_count >= max_attacks:
                    break
                
                logger.info("Executing red-team round", session=session_id, category=category, round=r)
                
                # 1. Generate Attack Prompt
                prompt = loop.run_until_complete(agent.generate_attack(category, r, previous_refusal))
                
                # 2. Fire at Target
                response = loop.run_until_complete(agent.call_target(prompt))
                attack_count += 1
                
                # 3. Evaluate Success
                is_success, severity, reasoning = agent.evaluate_success(response, category)
                
                # 4. Persist Round Result
                attack_record = RedTeamAttack(
                    session_id=session.id,
                    category=category,
                    severity=severity,
                    rounds=r,
                    is_successful=is_success,
                    encrypted_prompt=encrypt_content(prompt),
                    encrypted_response=encrypt_content(response),
                    judge_reasoning=reasoning
                )
                db.add(attack_record)
                
                if is_success:
                    success_count += 1
                    # Break the current category's rounds if successful
                    break
                else:
                    # Capture refusal for next round's escalation
                    previous_refusal = response
            
            # Checkpoint the session update periodically
            session.total_attacks = attack_count
            session.success_count = success_count
            db.commit()

        # Campaign Completed
        session.status = "COMPLETED"
        session.completed_at = datetime.utcnow()
        db.commit()
    
    except Exception as e:
        logger.error("Red-teaming campaign failed", session_id=session_id, error=str(e))
        if session:
             session.status = "FAILED"
             db.commit()
    finally:
        db.close()
