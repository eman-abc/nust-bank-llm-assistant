from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors

def create_mars_pdf(filename):
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter

    # Header
    c.setFillColor(colors.purple)
    c.rect(0, height - 100, width, 100, fill=1)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 24)
    c.drawString(50, height - 65, "NUST Bank - Mars Branch Policy")

    # Content
    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 150, "Circular No: SPACE/2050/001 - Red Planet Operations")
    
    c.setFont("Helvetica", 12)
    y = height - 180
    
    policies = [
        ("1. The Martian Gravity Savings Account", [
            "- Purpose: A high-yield account for residents of the Olympus Mons colony.",
            "- Profit Rate: 25.5% annually due to lower gravity overheads.",
            "- Minimum Balance: 100 Martian Credits (approx. 50,000 PKR).",
            "- Benefit: Free oxygen tank refills at any NUST Martian ATM."
        ]),
        ("2. Solar Flare Insurance Plan", [
            "- Purpose: Protection against digital asset loss during solar storms.",
            "- Coverage: Up to 5 Million PKR for data corruption events.",
            "- Eligibility: All customers using satellite-linked neural banking.",
            "- Premium: 0.1% of total account balance per lunar month."
        ]),
        ("3. Neural-Link Transaction Verification", [
            "- Purpose: Replacing biometric fingerprints with brain-wave patterns.",
            "- Hardware: Requires NUST-BrainLink v2.0 headband.",
            "- Daily Limit: Unlimited for neural-verified transfers between Mars and Earth.",
            "- Fee: 500 PKR for Inter-Planetary Network (IPN) transmission."
        ])
    ]

    for title, details in policies:
        c.setFont("Helvetica-Bold", 13)
        y -= 30
        c.drawString(50, y, title)
        c.setFont("Helvetica", 11)
        for line in details:
            y -= 20
            c.drawString(70, y, line)
        y -= 20

    # Footer
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(50, 50, "Security Classification: INTERGALACTIC - Authorized Personnel Only")
    
    c.save()
    print(f"Mars PDF created: {filename}")

if __name__ == "__main__":
    create_mars_pdf("NUST_Mars_Expansion_2050.pdf")
