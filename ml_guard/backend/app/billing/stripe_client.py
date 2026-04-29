import stripe
from app.core.config import settings
from typing import Optional, Dict, Any
import structlog

logger = structlog.get_logger()
stripe.api_key = settings.STRIPE_SECRET_KEY

class StripeClient:
    """
    Wrapper for Stripe API interactions including customer management,
    subscriptions, metered usage, and one-time checkout sessions.
    """
    
    @staticmethod
    def create_customer(org_id: str, email: str, name: str) -> str:
        try:
            customer = stripe.Customer.create(
                email=email,
                name=name,
                metadata={"org_id": org_id}
            )
            return customer.id
        except Exception as e:
            logger.error("STRIPE_CUSTOMER_CREATE_FAILED", error=str(e), org_id=org_id)
            raise

    @staticmethod
    def create_subscription(customer_id: str, price_id: str) -> Dict[str, Any]:
        try:
            subscription = stripe.Subscription.create(
                customer=customer_id,
                items=[{"price": price_id}],
                payment_behavior="default_incomplete",
                payment_settings={"save_default_payment_method": "on_subscription"},
                expand=["latest_invoice.payment_intent"],
            )
            return {
                "id": subscription.id,
                "client_secret": subscription.latest_invoice.payment_intent.client_secret
            }
        except Exception as e:
            logger.error("STRIPE_SUBSCRIPTION_CREATE_FAILED", error=str(e), customer_id=customer_id)
            raise

    @staticmethod
    def report_usage(stripe_customer_id: str, event_name: str, quantity: int):
        """
        Reports metered usage to Stripe using the MeterEvent API.
        """
        try:
            stripe.billing.MeterEvent.create(
                event_name=event_name,
                payload={
                    "value": str(quantity),
                    "stripe_customer_id": stripe_customer_id,
                },
            )
        except Exception as e:
            logger.error("STRIPE_USAGE_REPORT_FAILED", error=str(e), customer_id=stripe_customer_id)

    @staticmethod
    def create_compliance_checkout(customer_id: str, model_id: str, pack_name: str, success_url: str, cancel_url: str) -> str:
        try:
            session = stripe.checkout.Session.create(
                customer=customer_id,
                payment_method_types=["card"],
                line_items=[{
                    "price_data": {
                        "currency": "usd",
                        "product_data": {
                            "name": f"Compliance Certificate: {pack_name}",
                            "description": f"Official certification for model {model_id}",
                        },
                        "unit_amount": 50000, # $500.00
                    },
                    "quantity": 1,
                }],
                mode="payment",
                success_url=success_url,
                cancel_url=cancel_url,
                metadata={
                    "type": "compliance_certificate",
                    "model_id": model_id,
                    "pack_name": pack_name
                }
            )
            return session.url
        except Exception as e:
            logger.error("STRIPE_CHECKOUT_CREATE_FAILED", error=str(e), customer_id=customer_id)
            raise

    @staticmethod
    def construct_event(payload: bytes, sig_header: str):
        return stripe.Webhook.construct_event(
            payload, sig_header, settings.STRIPE_WEBHOOK_SECRET
        )
