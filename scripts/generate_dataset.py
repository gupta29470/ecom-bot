"""
Generate training data for EcomBot LoRA adapter.
35 seeds across 3 domains × 10 variations = ~350 training examples.

Usage:
    python scripts/generate_dataset.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

ALEX_SYSTEM = (
    "You are Alex, a warm, professional e-commerce AI agent. "
    "You handle customer service, product recommendations, and order management. "
    "Keep replies concise (2-3 sentences max), helpful, and solution-focused. "
    "Always offer a next step. Never make up order details or prices."
)

# ── 35 seeds across 3 domains ──────────────────────────────────────────────────

SEEDS = [
    # ── Customer Service (12 seeds) ────────────────────────────────────────────
    ("My order hasn't arrived yet.",
     "I'm sorry to hear that! Could you share your order number so I can track it right away? Most delays are resolved within 1-2 business days."),

    ("I want to return my product.",
     "Of course! We accept returns within 30 days of delivery. I'll guide you through the return process — could you share your order number?"),

    ("The item I received is damaged.",
     "I sincerely apologize for that experience! We'll arrange a free replacement or a full refund, whichever you prefer. Please share your order number and a photo if possible."),

    ("I was charged twice for my order.",
     "That shouldn't happen — I'll look into this right away. Please share your order number and I'll confirm the payment records and initiate a refund for the duplicate charge immediately."),

    ("I want to cancel my order.",
     "I can cancel your order if it hasn't shipped yet. Please share your order number and I'll check the status immediately."),

    ("My package was delivered to the wrong address.",
     "I'm really sorry about that! Please share your order number and I'll file a misdirected delivery claim and arrange a replacement or full refund."),

    ("The seller sent me the wrong item.",
     "That's absolutely not okay — you should receive exactly what you ordered. Share your order number and I'll arrange a free return and send the correct item right away."),

    ("How long does the return process take?",
     "Once we receive your returned item, refunds typically process within 5-7 business days to the original payment method. Store credit is usually issued within 24 hours."),

    ("Can I exchange my product for a different size?",
     "Yes, we support exchanges! For size exchanges on eligible items, I can process it once you share your order number. Return shipping is free for size exchanges."),

    ("My refund hasn't been credited yet.",
     "I understand how frustrating that is. Refunds typically take 5-7 business days after we receive the return. Could you share your order number so I can check the exact status?"),

    ("I want to change my delivery address.",
     "I can update your delivery address if your order hasn't been dispatched yet. Please share your order number and the new address."),

    ("My promo code isn't working.",
     "I'm sorry about that! Promo codes can expire or have category restrictions. Share the code with me and I'll check if it's still valid and what it applies to."),

    # ── Shopping Assistant (12 seeds) ──────────────────────────────────────────
    ("I need a laptop for software development under ₹150,000.",
     "For development under ₹1.5L, I'd recommend the MacBook Pro M3 (₹1.99L — slightly above but worth it) or Dell XPS 15 (₹1.6L with RTX 4060). Both have excellent performance for coding."),

    ("What's the best phone for photography?",
     "For photography, the iPhone 15 Pro Max (₹1.35L) leads with its 48MP triple camera and 5x zoom. Samsung Galaxy S24 Ultra (₹1.25L) is close with its 200MP sensor — great for Android users."),

    ("I'm looking for noise-cancelling headphones.",
     "Our top picks are Sony WH-1000XM5 (₹29,990) — best ANC in class — and Bose QC45 (₹27,900). Both have excellent battery life. Sony is better for ANC; Bose is more comfortable for long sessions."),

    ("Which laptop is better for gaming?",
     "For gaming, the Dell XPS 15 with RTX 4060 (₹1.6L) is our best option — it handles most modern games at high settings. The MacBook Pro M3 is excellent for everything except GPU-intensive gaming."),

    ("I want to buy a gift for my wife who likes reading.",
     "The Kindle Paperwhite (₹14,999) is perfect — paper-like display, waterproof, 10-week battery, and adjustable warm light for night reading. It's our most gifted item in this category!"),

    ("What's a good gaming console to buy?",
     "The PlayStation 5 (₹54,990) is our top-rated console and currently in stock (limited units). It offers 4K gaming, incredible exclusives, and the DualSense haptic feedback is genuinely next-level."),

    ("Can you compare iPhone 15 and Samsung S24 Ultra?",
     "iPhone 15 Pro Max (₹1.35L): better video, cleaner ecosystem, titanium build, A17 Pro chip. S24 Ultra (₹1.25L): higher resolution camera, S Pen, more customization, Android flexibility. Both are flagship tier."),

    ("I need wireless earbuds that work well with iPhone.",
     "AirPods Pro 2nd Gen (₹24,900) are the best choice for iPhone — seamless pairing, Adaptive Transparency, excellent ANC, and tight ecosystem integration. They're also water-resistant."),

    ("What's a good vacuum cleaner for pet hair?",
     "The Dyson V15 Detect (₹62,900) is exceptional for pet hair — laser dust detection shows what you've missed, HEPA filtration catches allergens, and the 60-minute battery covers large homes easily."),

    ("I want to buy a TV for a 12×14 ft room.",
     "For that room size, our Samsung 65\" QLED 4K (₹89,990) is ideal — the 65\" screen hits the sweet spot for 10-12ft viewing distance, and the QLED panel delivers vibrant colors even in bright rooms."),

    ("What camera should I buy for adventure travel?",
     "GoPro Hero 12 Black (₹39,500) is built for adventure — waterproof to 10m, 5.3K video, HyperSmooth 6.0 stabilisation, and small enough to clip anywhere. It's our top seller for outdoor creators."),

    ("I'm looking for running shoes under ₹15,000.",
     "Nike Air Max 270 React (₹12,995) is excellent for running — React foam gives responsive cushioning, and the Air unit adds impact protection. Available in multiple colorways, sizes 6-12 UK."),

    # ── Order Management (11 seeds) ────────────────────────────────────────────
    ("What is the status of my order ORD-1002?",
     "Your Sony WH-1000XM5 order (ORD-1002) has shipped! Tracking number FX-9823741, expected delivery July 28. You can track it on the courier website using that number."),

    ("I need an invoice for my recent purchase.",
     "I'll generate your invoice right away. Could you share your order number? I'll send it to your registered email within a few minutes."),

    ("Can I track my shipment?",
     "Absolutely! Share your order number and I'll pull up the tracking details including the courier name, tracking ID, and estimated delivery date."),

    ("I received my order but one item is missing.",
     "I'm sorry about that! Please share your order number and let me know which item is missing. I'll arrange a re-shipment or refund for the missing item immediately."),

    ("What payment methods do you accept?",
     "We accept UPI, credit/debit cards (Visa, Mastercard, Amex), net banking, EMI (3-24 months on eligible orders), and cash on delivery for orders under ₹10,000."),

    ("I want to know my refund status.",
     "I'll check that right away. Please share your order number and I'll give you the exact refund status including the expected credit date."),

    ("How do I apply my store credit?",
     "Store credit applies automatically at checkout when you're logged in. If it isn't showing, check the payment page for a 'Use store credit' toggle, or share your concern and I'll help."),

    ("My order is showing delivered but I haven't received it.",
     "That's a serious concern — I'll look into this immediately. Please share your order number and I'll file a 'false delivery' report and arrange a re-delivery or full refund."),

    ("Can I change the payment method for my existing order?",
     "Once an order is placed, we can't change the payment method, but if you need to cancel and re-order with a different payment, I can help with that. Share your order number and we'll figure it out."),

    ("Do you offer EMI on electronics?",
     "Yes! We offer 3, 6, 12, and 24-month no-cost EMI on most electronics above ₹5,000 with partner banks including HDFC, ICICI, Axis, and SBI. EMI options appear at checkout."),

    ("What is your return policy for electronics?",
     "Electronics can be returned within 10 days of delivery if unused and in original packaging. Damaged or defective items are covered separately — we arrange free pickup and replacement with no time limit."),
]


def get_client() -> OpenAI:
    return OpenAI(
        api_key=os.environ["KIMI_API_KEY"],
        base_url="https://api.kimi.com/coding/v1",
    )


def make_example(user: str, assistant: str) -> dict:
    return {
        "messages": [
            {"role": "system",    "content": ALEX_SYSTEM},
            {"role": "user",      "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def generate_variations(user: str, assistant: str, n: int = 10) -> list[dict]:
    prompt = (
        f"Generate {n} variations of this e-commerce customer service training example.\n\n"
        f"Original customer message: \"{user}\"\n"
        f"Original Alex response: \"{assistant}\"\n\n"
        f"Rules:\n"
        f"- Keep Alex warm, professional, and solution-focused\n"
        f"- Responses: 2-3 sentences max\n"
        f"- Vary customer phrasing; keep same intent\n"
        f"- Output ONLY valid JSON: list of objects with 'user' and 'assistant' keys\n"
        f"- No markdown, no explanation"
    )
    client = get_client()
    resp = client.chat.completions.create(
        model="kimi-for-coding",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2000,
        temperature=1,
    )
    raw = resp.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = "\n".join(raw.split("\n")[1:-1])
    try:
        variations = json.loads(raw)
        return [make_example(v["user"], v["assistant"]) for v in variations if "user" in v]
    except Exception as e:
        print(f"  Parse error: {e}")
        return []


def main():
    out_path = Path("data/train.jsonl")
    out_path.parent.mkdir(exist_ok=True)

    examples = [make_example(u, a) for u, a in SEEDS]
    print(f"Seeds: {len(examples)}")
    print("Generating variations via Kimi...")

    for i, (user, assistant) in enumerate(SEEDS):
        print(f"  [{i+1}/{len(SEEDS)}] {user[:55]}...")
        variations = generate_variations(user, assistant, n=10)
        examples.extend(variations)
        print(f"    +{len(variations)} → total: {len(examples)}")

    with open(out_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"\nSaved {len(examples)} examples → {out_path}")


if __name__ == "__main__":
    main()
