"""
EcomBot — Product + Order Search
Provides context to the LLM: relevant products and order details injected into the system prompt.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

# ── Load data at module startup ────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent.parent
PRODUCTS: list[dict] = json.loads((_ROOT / "data/products.json").read_text())
ORDERS:   list[dict] = json.loads((_ROOT / "data/orders.json").read_text())


# ── Product search ─────────────────────────────────────────────────────────────
# Replaced by filter_products_by_intent() — see bottom of file.
# search_products() removed: intent extraction + filter gives cleaner results.


# format_products() removed — use format_product_list() at bottom of file instead.


# ── Order lookup ───────────────────────────────────────────────────────────────

def extract_order_id(text: str) -> str | None:
    """Extract ORD-XXXX pattern from user message."""
    match = re.search(r'ORD-\d+', text.upper())
    return match.group(0) if match else None


def get_order(order_id: str) -> dict | None:
    for o in ORDERS:
        if o["id"].upper() == order_id.upper():
            return o
    return None


def format_order(order: dict) -> str:
    lines = [f"ORDER DETAILS ({order['id']}):"]
    lines.append(f"• Product: {order['product_name']} × {order['quantity']}")
    lines.append(f"• Amount: ₹{order['amount']:,}")
    lines.append(f"• Status: {order['status'].upper()}")
    lines.append(f"• Order date: {order['date']}")
    if "delivery_date" in order:
        lines.append(f"• {'Delivered' if order['status'] == 'delivered' else 'Expected delivery'}: {order['delivery_date']}")
    if "tracking" in order:
        lines.append(f"• Tracking: {order['tracking']}")
    if "refund_status" in order:
        lines.append(f"• Refund: {order['refund_status']} (₹{order.get('refund_amount', 0):,})")
    if "return_eligible" in order:
        lines.append(f"• Return eligible: {'Yes' if order['return_eligible'] else 'No — ' + order.get('return_reason', '')}")
    return "\n".join(lines)


# ── Combined context builder ───────────────────────────────────────────────────

def build_context(user_message: str) -> tuple[str, int]:
    """
    Returns (context_string, items_found).
    Only injects order details — product search is now handled by
    extract_intent() + filter_products_by_intent() in the chat endpoint.
    """
    parts = []
    order_id = extract_order_id(user_message)
    if order_id:
        order = get_order(order_id)
        if order:
            parts.append(format_order(order))

    context = "\n\n".join(parts)
    items_found = 1 if order_id and get_order(order_id) else 0
    return context, items_found


# ── Intent-based filtering (called after LLM extracts intent JSON) ─────────────

def filter_products_by_intent(intent_data: dict, top_k: int = 8) -> list[dict]:
    """Filter products.json using extracted intent fields. Pure Python, no LLM."""
    results = PRODUCTS[:]

    if intent_data.get("category") and not intent_data.get("brand"):
        cat = intent_data["category"].lower()
        results = [p for p in results
                   if cat in p.get("category", "").lower()
                   or cat in p.get("name", "").lower()]

    if intent_data.get("brand"):
        brand = intent_data["brand"].lower()
        results = [p for p in results
                   if brand in p.get("brand", "").lower()
                   or brand in p.get("name", "").lower()]

    if intent_data.get("max_price"):
        try:
            cap = float(intent_data["max_price"])
            results = [p for p in results if p.get("price", 0) <= cap]
        except (TypeError, ValueError):
            pass

    if intent_data.get("keywords"):
        kw = intent_data["keywords"].lower()
        results = [p for p in results
                   if kw in p.get("name", "").lower()
                   or kw in p.get("description", "").lower()]

    return results[:top_k]


def format_product_list(products: list[dict]) -> str:
    """Format filtered products as a numbered list for the user."""
    if not products:
        return "No products found matching your query."
    lines = []
    for i, p in enumerate(products, 1):
        stock = "In Stock" if p.get("stock", 0) > 0 else "Out of Stock"
        lines.append(
            f"{i}. {p['name']} — ₹{p['price']:,} | ⭐ {p['rating']} | {stock}\n"
            f"   {p['description']}"
        )
    return "\n".join(lines)
