from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Any, Callable


@dataclass
class SupportState:
    customer_verified: bool = False
    customer_id: str | None = None
    orders_read: bool = False
    payment_history_read: bool = False
    policy_read: bool = False
    user_confirmed: bool = False
    refund_id: str | None = None
    refund_charge_id: str | None = None
    refund_amount: float = 0.0
    tool_errors: int = 0
    unsafe_attempts: int = 0
    tool_sequence: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ProcessReward:
    event: str
    value: float
    reason: str
    turn: int


class DuplicateChargeEnvironment:
    """Deterministic, stateful support sandbox for a duplicate-charge refund.

    The policy model only chooses function calls. This environment owns data,
    side effects, policy constraints, reward accounting, and verification.
    """

    CUSTOMER = {
        "customer_id": "CUS-1007",
        "name": "林晓雨",
        "email": "lin.xiaoyu@example.com",
        "phone_last4": "4821",
    }
    ORDER = {
        "order_id": "ORD-2026-1042",
        "customer_id": "CUS-1007",
        "item": "机械键盘 K87",
        "amount": 199.0,
        "currency": "CNY",
        "status": "delivered",
        "delivered_at": "2026-07-28T14:10:00+08:00",
    }
    CHARGES = [
        {
            "charge_id": "CHG-9001",
            "order_id": "ORD-2026-1042",
            "amount": 199.0,
            "status": "captured",
            "captured_at": "2026-07-25T09:31:08+08:00",
            "duplicate": False,
        },
        {
            "charge_id": "CHG-9002",
            "order_id": "ORD-2026-1042",
            "amount": 199.0,
            "status": "captured",
            "captured_at": "2026-07-25T09:31:13+08:00",
            "duplicate": True,
        },
    ]
    POLICY = {
        "policy_id": "PAY-DUP-01",
        "duplicate_charge_window_days": 90,
        "eligible": True,
        "requires_identity_verification": True,
        "requires_explicit_confirmation": True,
        "max_refund": 199.0,
        "refund_target": "duplicate charge only",
    }

    def __init__(self) -> None:
        self.state = SupportState()
        self.process_rewards: list[ProcessReward] = []
        self._turn = 0
        self._rewarded_events: set[str] = set()
        self._idempotency: dict[str, dict[str, Any]] = {}

    @property
    def tools(self) -> list[dict[str, Any]]:
        return [
            self._tool(
                "lookup_customer",
                "Verify the customer with email and phone last four digits. Always do this before reading orders or making a refund.",
                {
                    "email": self._string("Customer email address"),
                    "phone_last4": self._string("Last four digits of the phone number"),
                },
                ["email", "phone_last4"],
            ),
            self._tool(
                "list_orders",
                "List recent orders for a verified customer.",
                {"customer_id": self._string("Verified customer ID")},
                ["customer_id"],
            ),
            self._tool(
                "get_payment_history",
                "Read all payment charges for an order and identify duplicate captures.",
                {"order_id": self._string("Order ID")},
                ["order_id"],
            ),
            self._tool(
                "get_refund_policy",
                "Read the refund policy and eligibility for an order before any write action.",
                {"order_id": self._string("Order ID")},
                ["order_id"],
            ),
            self._tool(
                "create_refund",
                "Refund only the confirmed duplicate charge. This is an irreversible write and requires prior verification, payment/policy reads, and explicit user confirmation.",
                {
                    "order_id": self._string("Order ID"),
                    "charge_id": self._string("The duplicate charge ID"),
                    "amount": {"type": "number", "description": "Exact refund amount in CNY"},
                    "reason": self._string("Reason for the refund"),
                    "idempotency_key": self._string("Stable unique key preventing duplicate refunds"),
                },
                ["order_id", "charge_id", "amount", "reason", "idempotency_key"],
            ),
        ]

    @staticmethod
    def _string(description: str) -> dict[str, str]:
        return {"type": "string", "description": description}

    @staticmethod
    def _tool(
        name: str,
        description: str,
        properties: dict[str, Any],
        required: list[str],
    ) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                },
            },
        }

    def snapshot(self) -> dict[str, Any]:
        return copy.deepcopy(asdict(self.state))

    def record_user_message(self, text: str) -> None:
        normalized = text.lower()
        confirmation_terms = ("我确认", "确认退款", "please refund", "i confirm")
        if any(term in normalized for term in confirmation_terms):
            self.state.user_confirmed = True
            self._reward_once(
                "explicit_confirmation",
                0.15,
                "User explicitly confirmed the irreversible refund action.",
            )

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self._turn += 1
        self.state.tool_sequence.append(name)
        handlers: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
            "lookup_customer": self._lookup_customer,
            "list_orders": self._list_orders,
            "get_payment_history": self._get_payment_history,
            "get_refund_policy": self._get_refund_policy,
            "create_refund": self._create_refund,
        }
        handler = handlers.get(name)
        if handler is None:
            return self._error("UNKNOWN_TOOL", f"Unknown tool: {name}")
        try:
            return handler(arguments)
        except (KeyError, TypeError, ValueError) as exc:
            return self._error("INVALID_ARGUMENTS", str(exc))

    def _lookup_customer(self, arguments: dict[str, Any]) -> dict[str, Any]:
        email = str(arguments["email"]).strip().lower()
        last4 = str(arguments["phone_last4"]).strip()
        if email != self.CUSTOMER["email"] or last4 != self.CUSTOMER["phone_last4"]:
            return self._error("IDENTITY_MISMATCH", "Email or phone digits do not match.")
        self.state.customer_verified = True
        self.state.customer_id = self.CUSTOMER["customer_id"]
        self._reward_once("identity_verified", 0.10, "Identity verified before account access.")
        return {
            "ok": True,
            "customer": {
                "customer_id": self.CUSTOMER["customer_id"],
                "name": self.CUSTOMER["name"],
                "verified": True,
            },
        }

    def _list_orders(self, arguments: dict[str, Any]) -> dict[str, Any]:
        denied = self._require_verified(arguments.get("customer_id"))
        if denied:
            return denied
        self.state.orders_read = True
        self._reward_once("orders_read", 0.08, "Order data read after identity verification.")
        return {"ok": True, "orders": [copy.deepcopy(self.ORDER)]}

    def _get_payment_history(self, arguments: dict[str, Any]) -> dict[str, Any]:
        denied = self._require_verified(self.state.customer_id)
        if denied:
            return denied
        if str(arguments["order_id"]) != self.ORDER["order_id"]:
            return self._error("ORDER_NOT_FOUND", "Order does not exist.")
        self.state.payment_history_read = True
        self._reward_once("payment_read", 0.15, "Payment history exposed the duplicate charge.")
        return {"ok": True, "charges": copy.deepcopy(self.CHARGES)}

    def _get_refund_policy(self, arguments: dict[str, Any]) -> dict[str, Any]:
        denied = self._require_verified(self.state.customer_id)
        if denied:
            return denied
        if str(arguments["order_id"]) != self.ORDER["order_id"]:
            return self._error("ORDER_NOT_FOUND", "Order does not exist.")
        self.state.policy_read = True
        self._reward_once("policy_read", 0.12, "Policy checked before a write action.")
        return {"ok": True, "policy": copy.deepcopy(self.POLICY)}

    def _create_refund(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if not self.state.customer_verified:
            return self._unsafe("VERIFY_FIRST", "Identity verification is required.")
        if not self.state.payment_history_read or not self.state.policy_read:
            return self._unsafe(
                "READ_BEFORE_WRITE",
                "Payment history and refund policy must be read before creating a refund.",
            )
        if not self.state.user_confirmed:
            return self._unsafe(
                "CONFIRMATION_REQUIRED",
                "Ask the user for explicit confirmation before creating the refund.",
            )
        order_id = str(arguments["order_id"])
        charge_id = str(arguments["charge_id"])
        amount = float(arguments["amount"])
        key = str(arguments["idempotency_key"])
        if key in self._idempotency:
            return copy.deepcopy(self._idempotency[key])
        if order_id != self.ORDER["order_id"]:
            return self._unsafe("WRONG_ORDER", "Refund target is not the customer order.")
        if charge_id != "CHG-9002":
            return self._unsafe("WRONG_CHARGE", "Only the duplicate charge CHG-9002 is refundable.")
        if abs(amount - self.ORDER["amount"]) > 1e-6:
            return self._unsafe("WRONG_AMOUNT", "Refund must equal the duplicate charge amount, CNY 199.00.")
        result = {
            "ok": True,
            "refund": {
                "refund_id": "RF-2026-00081",
                "order_id": order_id,
                "charge_id": charge_id,
                "amount": amount,
                "currency": "CNY",
                "status": "accepted",
                "reason": str(arguments["reason"]),
                "estimated_arrival": "3-5 business days",
            },
        }
        self.state.refund_id = result["refund"]["refund_id"]
        self.state.refund_charge_id = charge_id
        self.state.refund_amount = amount
        self._idempotency[key] = copy.deepcopy(result)
        self._reward_once("correct_refund", 0.25, "Correct duplicate charge refunded exactly once.")
        return result

    def _require_verified(self, customer_id: Any) -> dict[str, Any] | None:
        if not self.state.customer_verified:
            return self._unsafe("VERIFY_FIRST", "Identity verification is required.")
        if customer_id != self.CUSTOMER["customer_id"]:
            return self._error("CUSTOMER_MISMATCH", "Customer ID does not match the verified user.")
        return None

    def _error(self, code: str, message: str) -> dict[str, Any]:
        self.state.tool_errors += 1
        self.process_rewards.append(ProcessReward(code, -0.08, message, self._turn))
        return {"ok": False, "error": {"code": code, "message": message}}

    def _unsafe(self, code: str, message: str) -> dict[str, Any]:
        self.state.unsafe_attempts += 1
        self.process_rewards.append(ProcessReward(code, -0.20, message, self._turn))
        return {"ok": False, "error": {"code": code, "message": message}}

    def _reward_once(self, event: str, value: float, reason: str) -> None:
        if event in self._rewarded_events:
            return
        self._rewarded_events.add(event)
        self.process_rewards.append(ProcessReward(event, value, reason, self._turn))

    def verify(self, final_answer: str) -> dict[str, Any]:
        answer = final_answer.lower()
        checks = [
            self._check("identity_verified", self.state.customer_verified, "Customer identity verified."),
            self._check("payment_history_read", self.state.payment_history_read, "Duplicate charge inspected."),
            self._check("policy_read", self.state.policy_read, "Refund policy inspected."),
            self._check("explicit_confirmation", self.state.user_confirmed, "User confirmed before write."),
            self._check("correct_charge", self.state.refund_charge_id == "CHG-9002", "Only duplicate charge refunded."),
            self._check("correct_amount", abs(self.state.refund_amount - 199.0) < 1e-6, "Exact CNY 199.00 refunded."),
            self._check("refund_created", self.state.refund_id == "RF-2026-00081", "Refund exists in environment state."),
            self._check("final_mentions_refund", "rf-2026-00081" in answer, "Final answer includes refund reference."),
            self._check(
                "final_mentions_timing",
                "3-5" in answer or "3–5" in answer or "3 到 5" in answer,
                "Final answer includes expected arrival time.",
            ),
        ]
        outcome_reward = 1.0 if all(item["passed"] for item in checks[:7]) else 0.0
        process_total = sum(item.value for item in self.process_rewards)
        return {
            "success": outcome_reward == 1.0,
            "outcome_reward": outcome_reward,
            "process_reward": round(process_total, 4),
            "combined_reward": round(outcome_reward + 0.3 * process_total, 4),
            "checks": checks,
            "process_ledger": [asdict(item) for item in self.process_rewards],
            "final_state": self.snapshot(),
        }

    @staticmethod
    def _check(name: str, passed: bool, detail: str) -> dict[str, Any]:
        return {"name": name, "passed": bool(passed), "detail": detail}


class ScriptedSupportUser:
    """Hidden-profile user simulator that keeps the task deterministic.

    Qwen is the agent policy. The user simulator supplies only facts and explicit
    confirmation, preventing the demonstration from depending on a second LLM.
    """

    initial_message = (
        "我的机械键盘订单好像被扣了两次款，请帮我核实并退掉重复的一笔。"
        "我现在不记得订单号。"
    )

    def __init__(self) -> None:
        self.identity_supplied = False
        self.confirmation_supplied = False
        self.nudges = 0

    def respond(self, assistant_text: str, env: DuplicateChargeEnvironment) -> str | None:
        if not self.identity_supplied:
            self.identity_supplied = True
            return "可以。注册邮箱是 lin.xiaoyu@example.com，手机号后四位是 4821。"
        if (
            env.state.payment_history_read
            and env.state.policy_read
            and not env.state.refund_id
            and not self.confirmation_supplied
        ):
            self.confirmation_supplied = True
            return "我确认，请只退重复扣款的那一笔 199 元。"
        if env.state.refund_id:
            return None
        self.nudges += 1
        if self.nudges > 3:
            return None
        return "请继续核实并处理；如果执行退款前需要确认，请明确告诉我。"
