from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors

def create_policy_pdf(filename):
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter

    # Header
    c.setFillColor(colors.darkblue)
    c.rect(0, height - 100, width, 100, fill=1)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 24)
    c.drawString(50, height - 65, "NUST Bank - 2026 Policy Circular")

    # Content
    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 150, "Circular No: NUST/2026/001 - New Product Launch")
    
    c.setFont("Helvetica", 12)
    y = height - 180
    
    policies = [
        ("1. NUST-Green EV Financing Scheme", [
            "- Purpose: Financing for Electric Vehicles (EVs) for individual customers.",
            "- Benefit: 1.5% discount on prevailing markup rates.",
            "- Eligibility: All salaried and self-employed individuals with 3+ years' track record.",
            "- Max Limit: Up to 10 Million PKR."
        ]),
        ("2. NUST-Youth Digital Bond (Age 18-25)", [
            "- Purpose: Focused saving instrument for the digital generation.",
            "- Profit Rate: Fixed 14.5% per annum.",
            "- Minimum Investment: 5,000 PKR.",
            "- Maturity Period: 1 Year, 3 Year, and 5 Year options available.",
            "- Withdrawal: Zero penalty for one emergency withdrawal per year."
        ]),
        ("3. QR-Cash Express (Cardless)", [
            "- Purpose: Enhanced cardless cash withdrawals via NUST App.",
            "- Daily Limit: Increased to 500,000 PKR per day.",
            "- Transaction Fee: 0 PKR for NUST-to-NUST QR transactions.",
            "- Security: Uses dynamic time-bound QR codes with biometric verification."
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
    c.drawString(50, 50, "Secret: For Internal Use Only - NUST Bank Compliance Department 2026")
    
    c.save()
    print(f"Policy PDF created: {filename}")

if __name__ == "__main__":
    create_policy_pdf("NUST_Bank_2026_Upgrades.pdf")
