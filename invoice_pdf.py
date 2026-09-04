"""
Dynamic PDF invoice generation using ReportLab.

Kept as its own module (rather than inline in main.py) purely for
readability — main.py imports generate_invoice_pdf_bytes() and streams
the result back over HTTP.
"""

import io
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Table,
    TableStyle,
    Paragraph,
    Spacer,
    PageBreak,
)


def generate_invoice_pdf_bytes(invoice) -> bytes:
    """
    Builds a simple, clean invoice PDF for the given Invoice ORM object
    (with its .client relationship loaded) and returns the raw PDF bytes.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        title=f"Invoice {invoice.invoice_number}",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "InvoiceTitle", parent=styles["Heading1"], textColor=colors.HexColor("#065f46")
    )
    label_style = ParagraphStyle(
        "Label", parent=styles["Normal"], textColor=colors.HexColor("#64748b"), fontSize=9
    )
    normal = styles["Normal"]

    status_colors = {
        "PENDING": colors.HexColor("#b45309"),
        "PARTIALLY_PAID": colors.HexColor("#1d4ed8"),
        "PAID": colors.HexColor("#047857"),
        "DISPUTED": colors.HexColor("#b91c1c"),
    }
    status_value = invoice.status.value if hasattr(invoice.status, "value") else str(invoice.status)

    elements = []

    elements.append(Paragraph("LedgerRecover AI", title_style))
    elements.append(Paragraph("Autonomous B2B Trade Collection", label_style))
    elements.append(Spacer(1, 14 * mm))

    header_table_data = [
        [
            Paragraph(f"<b>Invoice {invoice.invoice_number}</b>", styles["Heading2"]),
            Paragraph(
                f'<font color="{status_colors.get(status_value, "#334155")}"><b>{status_value}</b></font>',
                normal,
            ),
        ]
    ]
    header_table = Table(header_table_data, colWidths=[110 * mm, 50 * mm])
    header_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (1, 0), (1, 0), "RIGHT"),
    ]))
    elements.append(header_table)
    elements.append(Spacer(1, 8 * mm))

    bill_to = [
        Paragraph("<b>Bill To</b>", label_style),
        Paragraph(invoice.client.business_name, styles["Heading3"]),
        Paragraph(invoice.client.name, normal),
        Paragraph(invoice.client.phone_number, normal),
    ]
    for p in bill_to:
        elements.append(p)
    elements.append(Spacer(1, 8 * mm))

    meta_data = [
        ["Invoice Date", invoice.created_at.strftime("%d %b %Y")],
        ["Due Date", invoice.due_date.strftime("%d %b %Y")],
        ["Invoice Number", invoice.invoice_number],
    ]
    meta_table = Table(meta_data, colWidths=[45 * mm, 100 * mm])
    meta_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#64748b")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    elements.append(meta_table)
    elements.append(Spacer(1, 10 * mm))


    # Payment history breakdown — only shown once there's more than one
    # installment; a single payment is already fully represented by the
    # "Amount Paid" line below, so a one-row table would just be noise.
    payments_so_far = sorted(invoice.payments, key=lambda p: p.paid_at)
    # Combined PAY ALL transactions are stored once as a provider payment and
    # distributed through PaymentAllocation rows. Show allocations here too
    # when this invoice was not the transaction's anchor invoice.
    allocation_rows = [
        (a.payment.paid_at, a.amount_allocated)
        for a in getattr(invoice, "payment_allocations", [])
        if a.payment and a.payment.invoice_id != invoice.id
    ]
    history_events = [(p.paid_at, p.amount_paid) for p in payments_so_far] + allocation_rows
    history_events.sort(key=lambda x: x[0])
    if len(history_events) > 1:
        elements.append(Paragraph("<b>Payment History</b>", label_style))
        elements.append(Spacer(1, 3 * mm))
        history_rows = [["Date", "Amount"]] + [
            [dt.strftime("%d %b %Y"), f"Rs. {amount:,.2f}"]
            for dt, amount in history_events
        ]
        history_table = Table(history_rows, colWidths=[110 * mm, 50 * mm])
        history_table.setStyle(TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#475569")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ALIGN", (1, 0), (1, -1), "RIGHT"),
            ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.HexColor("#cbd5e1")),
            ("LINEBELOW", (0, 1), (-1, -1), 0.25, colors.HexColor("#e2e8f0")),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        elements.append(history_table)
        elements.append(Spacer(1, 6 * mm))

    amounts_data = [
        ["Total Bill Amount", f"Rs. {invoice.total_amount:,.2f}"],
        ["Amount Paid", f"Rs. {invoice.paid_amount:,.2f}"],
        ["Balance Due", f"Rs. {invoice.balance_amount:,.2f}"],
    ]
    amounts_table = Table(amounts_data, colWidths=[110 * mm, 50 * mm])
    amounts_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ALIGN", (1, 0), (1, -1), "RIGHT"),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.HexColor("#e2e8f0")),
        ("LINEBELOW", (0, 1), (-1, 1), 0.5, colors.HexColor("#e2e8f0")),
        ("LINEABOVE", (0, 2), (-1, 2), 1, colors.HexColor("#065f46")),
        ("FONTNAME", (0, 2), (-1, 2), "Helvetica-Bold"),
        ("FONTSIZE", (0, 2), (-1, 2), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(amounts_table)
    elements.append(Spacer(1, 16 * mm))

    elements.append(Paragraph(
        "This is a system-generated invoice from LedgerRecover AI. "
        "For questions about this bill, reply on WhatsApp to the number this invoice was sent from.",
        label_style,
    ))

    doc.build(elements)
    return buffer.getvalue()


# --------------------------------------------------------------------------
# NEW (Feature 5): Zero-Balance Clearance Receipt PDF
# --------------------------------------------------------------------------

def generate_receipt_pdf_bytes(invoice) -> bytes:
    """
    Builds a "Payment Receipt / Zero-Balance Clearance" PDF for an invoice
    whose balance_amount has hit 0.0. Expects invoice.client and
    invoice.payments (ordered by paid_at) to be loaded/loadable via the
    SQLAlchemy relationships already defined on the Invoice model.

    Unlike generate_invoice_pdf_bytes, this lists the FULL payment history
    for the invoice (not just the final balance) — real khatabook-style
    apps keep the full trail on a cleared bill for audit/GST purposes
    rather than only showing the last payment that zeroed it out.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        title=f"Receipt - {invoice.invoice_number}",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReceiptTitle", parent=styles["Heading1"], textColor=colors.HexColor("#047857")
    )
    label_style = ParagraphStyle(
        "Label", parent=styles["Normal"], textColor=colors.HexColor("#64748b"), fontSize=9
    )
    normal = styles["Normal"]

    elements = []

    elements.append(Paragraph("LedgerRecover AI", title_style))
    elements.append(Paragraph("Payment Receipt / Zero-Balance Clearance", label_style))
    elements.append(Spacer(1, 14 * mm))

    header_table_data = [
        [
            Paragraph(f"<b>Invoice {invoice.invoice_number}</b>", styles["Heading2"]),
            Paragraph(
                '<font color="#047857"><b>PAID &amp; CLEARED</b></font>',
                normal,
            ),
        ]
    ]
    header_table = Table(header_table_data, colWidths=[110 * mm, 50 * mm])
    header_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (1, 0), (1, 0), "RIGHT"),
    ]))
    elements.append(header_table)
    elements.append(Spacer(1, 8 * mm))

    bill_to = [
        Paragraph("<b>Received From</b>", label_style),
        Paragraph(invoice.client.business_name, styles["Heading3"]),
        Paragraph(invoice.client.name, normal),
        Paragraph(invoice.client.phone_number, normal),
    ]
    for p in bill_to:
        elements.append(p)
    elements.append(Spacer(1, 8 * mm))

    meta_data = [
        ["Invoice Date", invoice.created_at.strftime("%d %b %Y")],
        ["Original Due Date", invoice.due_date.strftime("%d %b %Y")],
        ["Invoice Number", invoice.invoice_number],
        ["Cleared On", datetime.utcnow().strftime("%d %b %Y")] if False else
        ["Invoice Total", f"Rs. {invoice.total_amount:,.2f}"],
    ]
    meta_table = Table(meta_data, colWidths=[45 * mm, 100 * mm])
    meta_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#64748b")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    elements.append(meta_table)
    elements.append(Spacer(1, 10 * mm))

    # Payment history table — every PaymentRecord tied to this invoice,
    # oldest first, so the receipt reads as a full audit trail.
    payments = sorted(invoice.payments, key=lambda p: p.paid_at)
    history_events = [(p.paid_at, p.razorpay_payment_id, p.amount_paid) for p in payments]
    history_events += [
        (a.payment.paid_at, a.payment.razorpay_payment_id, a.amount_allocated)
        for a in getattr(invoice, "payment_allocations", [])
        if a.payment and a.payment.invoice_id != invoice.id
    ]
    history_events.sort(key=lambda x: x[0])
    history_rows = [["Date", "Payment Reference", "Amount"]]
    for paid_at, payment_ref, display_amount in history_events:
        ref = payment_ref
        # Trim long Razorpay/internal ids so the column doesn't wrap ugly.
        display_ref = ref if len(ref) <= 28 else ref[:25] + "..."
        history_rows.append([
            paid_at.strftime("%d %b %Y"),
            display_ref,
            f"Rs. {display_amount:,.2f}",
        ])

    history_table = Table(history_rows, colWidths=[35 * mm, 85 * mm, 40 * mm])
    history_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#475569")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (2, 0), (2, -1), "RIGHT"),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.HexColor("#cbd5e1")),
        ("LINEBELOW", (0, 1), (-1, -1), 0.25, colors.HexColor("#e2e8f0")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    elements.append(history_table)
    elements.append(Spacer(1, 8 * mm))

    totals_data = [
        ["Total Bill Amount", f"Rs. {invoice.total_amount:,.2f}"],
        ["Total Amount Paid", f"Rs. {invoice.paid_amount:,.2f}"],
        ["Balance Remaining", "Rs. 0.00"],
    ]
    totals_table = Table(totals_data, colWidths=[110 * mm, 50 * mm])
    totals_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ALIGN", (1, 0), (1, -1), "RIGHT"),
        ("LINEABOVE", (0, 2), (-1, 2), 1, colors.HexColor("#047857")),
        ("FONTNAME", (0, 2), (-1, 2), "Helvetica-Bold"),
        ("FONTSIZE", (0, 2), (-1, 2), 12),
        ("TEXTCOLOR", (0, 2), (-1, 2), colors.HexColor("#047857")),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(totals_table)
    elements.append(Spacer(1, 16 * mm))

    elements.append(Paragraph(
        "This is a system-generated payment receipt from LedgerRecover AI, retained for "
        "audit and GST compliance purposes. For questions about this bill, reply on "
        "WhatsApp to the number this invoice was sent from.",
        label_style,
    ))

    doc.build(elements)
    return buffer.getvalue()

# --------------------------------------------------------------------------
# NEW: Combined multi-invoice PDF — one document, one page per invoice.
# --------------------------------------------------------------------------

def generate_combined_invoices_pdf_bytes(invoices) -> bytes:
    """
    Builds ONE PDF containing every invoice in `invoices`, each on its own
    page (via PageBreak), in the order given. Used when a buyer with more
    than one open invoice explicitly asks for "all"/"both" invoices as a
    single combined document, instead of separate download links per
    invoice. Each page reuses the exact same layout as the single-invoice
    PDF (generate_invoice_pdf_bytes) so it looks identical to what the buyer
    would get by downloading each invoice one at a time — just stapled
    together — plus a short cover section summarizing the total due.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        title="Combined Invoices — LedgerRecover AI",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "CombinedTitle", parent=styles["Heading1"], textColor=colors.HexColor("#065f46")
    )
    label_style = ParagraphStyle(
        "Label", parent=styles["Normal"], textColor=colors.HexColor("#64748b"), fontSize=9
    )
    normal = styles["Normal"]

    status_colors = {
        "PENDING": colors.HexColor("#b45309"),
        "PARTIALLY_PAID": colors.HexColor("#1d4ed8"),
        "PAID": colors.HexColor("#047857"),
        "DISPUTED": colors.HexColor("#b91c1c"),
    }

    elements = []

    # --- Cover / summary page -------------------------------------------
    elements.append(Paragraph("LedgerRecover AI", title_style))
    elements.append(Paragraph("Combined Invoice Statement", label_style))
    elements.append(Spacer(1, 10 * mm))

    if invoices:
        client = invoices[0].client
        elements.append(Paragraph(f"<b>{client.business_name}</b>", styles["Heading3"]))
        elements.append(Paragraph(f"{client.name} · {client.phone_number}", normal))
    elements.append(Spacer(1, 8 * mm))

    total_due = sum(inv.balance_amount for inv in invoices)
    summary_rows = [["Invoice", "Due Date", "Status", "Balance"]]
    for inv in invoices:
        status_value = inv.status.value if hasattr(inv.status, "value") else str(inv.status)
        summary_rows.append([
            inv.invoice_number,
            inv.due_date.strftime("%d %b %Y"),
            status_value,
            f"Rs. {inv.balance_amount:,.2f}",
        ])
    summary_rows.append(["", "", "Total Due", f"Rs. {total_due:,.2f}"])

    summary_table = Table(summary_rows, colWidths=[45 * mm, 35 * mm, 40 * mm, 40 * mm])
    summary_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#475569")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (3, 0), (3, -1), "RIGHT"),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.HexColor("#cbd5e1")),
        ("LINEBELOW", (0, 1), (-1, -2), 0.25, colors.HexColor("#e2e8f0")),
        ("LINEABOVE", (0, -1), (-1, -1), 1, colors.HexColor("#065f46")),
        ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    elements.append(summary_table)
    elements.append(Spacer(1, 6 * mm))
    elements.append(Paragraph(
        f"This document contains the full detail for all {len(invoices)} invoice(s) listed "
        "above, one per page, following this summary.",
        label_style,
    ))
    elements.append(PageBreak())

    # --- One full detail page per invoice ---------------------------------
    for idx, invoice in enumerate(invoices):
        status_value = invoice.status.value if hasattr(invoice.status, "value") else str(invoice.status)

        elements.append(Paragraph("LedgerRecover AI", title_style))
        elements.append(Paragraph("Autonomous B2B Trade Collection", label_style))
        elements.append(Spacer(1, 14 * mm))

        header_table = Table([[
            Paragraph(f"<b>Invoice {invoice.invoice_number}</b>", styles["Heading2"]),
            Paragraph(
                f'<font color="{status_colors.get(status_value, "#334155")}"><b>{status_value}</b></font>',
                normal,
            ),
        ]], colWidths=[110 * mm, 50 * mm])
        header_table.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ALIGN", (1, 0), (1, 0), "RIGHT"),
        ]))
        elements.append(header_table)
        elements.append(Spacer(1, 8 * mm))

        elements.append(Paragraph("<b>Bill To</b>", label_style))
        elements.append(Paragraph(invoice.client.business_name, styles["Heading3"]))
        elements.append(Paragraph(invoice.client.name, normal))
        elements.append(Paragraph(invoice.client.phone_number, normal))
        elements.append(Spacer(1, 8 * mm))

        meta_table = Table([
            ["Invoice Date", invoice.created_at.strftime("%d %b %Y")],
            ["Due Date", invoice.due_date.strftime("%d %b %Y")],
            ["Invoice Number", invoice.invoice_number],
        ], colWidths=[45 * mm, 100 * mm])
        meta_table.setStyle(TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#64748b")),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        elements.append(meta_table)
        elements.append(Spacer(1, 10 * mm))

        payments_so_far = sorted(invoice.payments, key=lambda p: p.paid_at)
        if len(payments_so_far) > 1:
            elements.append(Paragraph("<b>Payment History</b>", label_style))
            elements.append(Spacer(1, 3 * mm))
            history_rows = [["Date", "Amount"]] + [
                [p.paid_at.strftime("%d %b %Y"), f"Rs. {p.amount_paid:,.2f}"]
                for p in payments_so_far
            ]
            history_table = Table(history_rows, colWidths=[110 * mm, 50 * mm])
            history_table.setStyle(TableStyle([
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#475569")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ALIGN", (1, 0), (1, -1), "RIGHT"),
                ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.HexColor("#cbd5e1")),
                ("LINEBELOW", (0, 1), (-1, -1), 0.25, colors.HexColor("#e2e8f0")),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]))
            elements.append(history_table)
            elements.append(Spacer(1, 6 * mm))

        amounts_table = Table([
            ["Total Bill Amount", f"Rs. {invoice.total_amount:,.2f}"],
            ["Amount Paid", f"Rs. {invoice.paid_amount:,.2f}"],
            ["Balance Due", f"Rs. {invoice.balance_amount:,.2f}"],
        ], colWidths=[110 * mm, 50 * mm])
        amounts_table.setStyle(TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("ALIGN", (1, 0), (1, -1), "RIGHT"),
            ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.HexColor("#e2e8f0")),
            ("LINEBELOW", (0, 1), (-1, 1), 0.5, colors.HexColor("#e2e8f0")),
            ("LINEABOVE", (0, 2), (-1, 2), 1, colors.HexColor("#065f46")),
            ("FONTNAME", (0, 2), (-1, 2), "Helvetica-Bold"),
            ("FONTSIZE", (0, 2), (-1, 2), 12),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        elements.append(amounts_table)

        if idx < len(invoices) - 1:
            elements.append(PageBreak())

    doc.build(elements)
    return buffer.getvalue()