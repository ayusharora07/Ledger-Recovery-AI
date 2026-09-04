import enum
from datetime import datetime
from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime,
    Enum,
    ForeignKey,
    Text,
    Boolean,
)
from sqlalchemy.orm import relationship
from database import Base


class InvoiceStatus(str, enum.Enum):
    PENDING = "PENDING"
    PARTIALLY_PAID = "PARTIALLY_PAID"
    PAID = "PAID"
    DISPUTED = "DISPUTED"


class AuditStatus(str, enum.Enum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"


class CollectionCaseStatus(str, enum.Enum):
    NOT_DUE = "NOT_DUE"
    DUE = "DUE"
    OVERDUE = "OVERDUE"
    CONTACTED = "CONTACTED"
    PROMISED = "PROMISED"
    PAYMENT_PENDING = "PAYMENT_PENDING"
    PARTIALLY_PAID = "PARTIALLY_PAID"
    DISPUTED = "DISPUTED"
    ESCALATED = "ESCALATED"
    PAID = "PAID"


class PromiseStatus(str, enum.Enum):
    ACTIVE = "ACTIVE"
    FULFILLED = "FULFILLED"
    BROKEN = "BROKEN"
    CANCELLED = "CANCELLED"


class CollectionActionStatus(str, enum.Enum):
    SCHEDULED = "SCHEDULED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    SKIPPED = "SKIPPED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class Client(Base):
    __tablename__ = "clients"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    phone_number = Column(String, unique=True, index=True, nullable=False)
    business_name = Column(String, nullable=False)

    invoices = relationship(
        "Invoice", back_populates="client", cascade="all, delete-orphan"
    )


class Invoice(Base):
    __tablename__ = "invoices"

    id = Column(Integer, primary_key=True, index=True)
    invoice_number = Column(String, unique=True, index=True, nullable=False)
    client_id = Column(Integer, ForeignKey("clients.id"), nullable=False)

    total_amount = Column(Float, nullable=False)
    paid_amount = Column(Float, default=0.0, nullable=False)
    # balance_amount is a persisted, application-enforced derived field
    # (total_amount - paid_amount). It is NEVER set directly from user
    # or LLM input — only recomputed by trusted application code
    # (see agent_engine.py / main.py in later steps).
    balance_amount = Column(Float, nullable=False)

    status = Column(Enum(InvoiceStatus), default=InvoiceStatus.PENDING, nullable=False)
    due_date = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    client = relationship("Client", back_populates="invoices")
    payments = relationship(
        "PaymentRecord", back_populates="invoice", cascade="all, delete-orphan"
    )
    audit_logs = relationship(
        "AuditLog", back_populates="invoice", cascade="all, delete-orphan"
    )
    payment_allocations = relationship(
        "PaymentAllocation", back_populates="invoice", cascade="all, delete-orphan"
    )
    promises = relationship(
        "PaymentPromise", back_populates="invoice", cascade="all, delete-orphan"
    )
    collection_actions = relationship(
        "CollectionAction", back_populates="invoice", cascade="all, delete-orphan"
    )


class PaymentRecord(Base):
    __tablename__ = "payment_records"

    id = Column(Integer, primary_key=True, index=True)
    invoice_id = Column(Integer, ForeignKey("invoices.id"), nullable=False)
    razorpay_payment_id = Column(String, unique=True, nullable=False)
    razorpay_payment_link_id = Column(String, nullable=False)
    amount_paid = Column(Float, nullable=False)
    paid_at = Column(DateTime, default=datetime.utcnow)

    invoice = relationship("Invoice", back_populates="payments")
    allocations = relationship("PaymentAllocation", back_populates="payment", cascade="all, delete-orphan")


class PaymentAllocation(Base):
    __tablename__ = "payment_allocations"

    id = Column(Integer, primary_key=True, index=True)
    payment_id = Column(Integer, ForeignKey("payment_records.id"), nullable=False, index=True)
    invoice_id = Column(Integer, ForeignKey("invoices.id"), nullable=False, index=True)
    amount_allocated = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    payment = relationship("PaymentRecord", back_populates="allocations")
    invoice = relationship("Invoice", back_populates="payment_allocations")


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    invoice_id = Column(Integer, ForeignKey("invoices.id"), nullable=True)

    incoming_message = Column(Text, nullable=False)
    detected_intent = Column(String, nullable=False)
    action_taken = Column(String, nullable=False)
    status = Column(Enum(AuditStatus), default=AuditStatus.SUCCESS, nullable=False)
    execution_payload = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

    invoice = relationship("Invoice", back_populates="audit_logs")

class CollectionCase(Base):
    __tablename__ = "collection_cases"

    id = Column(Integer, primary_key=True, index=True)
    client_id = Column(Integer, ForeignKey("clients.id"), unique=True, nullable=False)
    status = Column(Enum(CollectionCaseStatus), default=CollectionCaseStatus.NOT_DUE, nullable=False)
    priority_score = Column(Float, default=0.0, nullable=False)
    next_action_at = Column(DateTime, nullable=True, index=True)
    last_contact_at = Column(DateTime, nullable=True)
    last_customer_response_at = Column(DateTime, nullable=True)
    escalation_reason = Column(Text, nullable=True)
    autonomous_enabled = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    client = relationship("Client")
    actions = relationship("CollectionAction", back_populates="case", cascade="all, delete-orphan")
    promises = relationship("PaymentPromise", back_populates="case", cascade="all, delete-orphan")


class PaymentPromise(Base):
    __tablename__ = "payment_promises"

    id = Column(Integer, primary_key=True, index=True)
    client_id = Column(Integer, ForeignKey("clients.id"), nullable=False, index=True)
    invoice_id = Column(Integer, ForeignKey("invoices.id"), nullable=True, index=True)
    case_id = Column(Integer, ForeignKey("collection_cases.id"), nullable=True, index=True)
    promised_amount = Column(Float, nullable=True)
    promised_date = Column(DateTime, nullable=False, index=True)
    status = Column(Enum(PromiseStatus), default=PromiseStatus.ACTIVE, nullable=False)
    source_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    fulfilled_at = Column(DateTime, nullable=True)
    broken_at = Column(DateTime, nullable=True)

    client = relationship("Client")
    invoice = relationship("Invoice", back_populates="promises")
    case = relationship("CollectionCase", back_populates="promises")


class CollectionAction(Base):
    __tablename__ = "collection_actions"

    id = Column(Integer, primary_key=True, index=True)
    client_id = Column(Integer, ForeignKey("clients.id"), nullable=False, index=True)
    invoice_id = Column(Integer, ForeignKey("invoices.id"), nullable=True, index=True)
    case_id = Column(Integer, ForeignKey("collection_cases.id"), nullable=True, index=True)
    action_type = Column(String, nullable=False)
    status = Column(Enum(CollectionActionStatus), default=CollectionActionStatus.SCHEDULED, nullable=False)
    scheduled_at = Column(DateTime, nullable=False, index=True)
    executed_at = Column(DateTime, nullable=True)
    attempt_count = Column(Integer, default=0, nullable=False)
    reason = Column(Text, nullable=True)
    result = Column(Text, nullable=True)
    dedupe_key = Column(String, unique=True, nullable=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    client = relationship("Client")
    invoice = relationship("Invoice", back_populates="collection_actions")
    case = relationship("CollectionCase", back_populates="actions")


class Communication(Base):
    __tablename__ = "communications"

    id = Column(Integer, primary_key=True, index=True)
    client_id = Column(Integer, ForeignKey("clients.id"), nullable=False, index=True)
    invoice_id = Column(Integer, ForeignKey("invoices.id"), nullable=True, index=True)
    direction = Column(String, nullable=False)  # INBOUND / OUTBOUND
    channel = Column(String, default="SIMULATOR", nullable=False)
    message = Column(Text, nullable=False)
    provider_message_id = Column(String, nullable=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    client = relationship("Client")
    invoice = relationship("Invoice")
