# backend/db/init_db.py
import sqlite3


def init_db():
    conn = sqlite3.connect("backend/db/operators.db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS operators (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            ward_id INTEGER NOT NULL
        )
    """)

    operators = [
        ("Amit Sharma", "amit.sharma.a@bmc.gov.in", "amitA@123", 1),
        ("Sneha Patel", "sneha.patel.b@bmc.gov.in", "snehaB@123", 2),
        ("Ravi Iyer", "ravi.iyer.c@bmc.gov.in", "raviC@123", 3),
        ("Pooja Mehta", "pooja.mehta.d@bmc.gov.in", "poojaD@123", 4),
        ("Rajesh Nair", "rajesh.nair.e@bmc.gov.in", "rajeshE@123", 5),
        ("Meena Joshi", "meena.joshi.fn@bmc.gov.in", "meenaFN@123", 6),
        ("Arun Verma", "arun.verma.fs@bmc.gov.in", "arunFS@123", 7),
        ("Kiran Rao", "kiran.rao.gn@bmc.gov.in", "kiranGN@123", 8),
        ("Deepa Kulkarni", "deepa.kulkarni.gs@bmc.gov.in", "deepaGS@123", 9),
        ("Suresh Yadav", "suresh.yadav.he@bmc.gov.in", "sureshHE@123", 10),
        ("Nita Deshmukh", "nita.deshmukh.hw@bmc.gov.in", "nitaHW@123", 11),
        ("Mahesh Gupta", "mahesh.gupta.ke@bmc.gov.in", "maheshKE@123", 12),
        ("Anjali Pillai", "anjali.pillai.kw@bmc.gov.in", "anjaliKW@123", 13),
        ("Rakesh Singh", "rakesh.singh.l@bmc.gov.in", "rakeshL@123", 14),
        ("Sunita Ghosh", "sunita.ghosh.me@bmc.gov.in", "sunitaME@123", 15),
        ("Vikram Das", "vikram.das.mw@bmc.gov.in", "vikramMW@123", 16),
        ("Priya Banerjee", "priya.banerjee.n@bmc.gov.in", "priyaN@123", 17),
        ("Ankit Tiwari", "ankit.tiwari.pn@bmc.gov.in", "ankitPN@123", 18),
        ("Kavita Rao", "kavita.rao.ps@bmc.gov.in", "kavitaPS@123", 19),
        ("Rohit Shetty", "rohit.shetty.rn@bmc.gov.in", "rohitRN@123", 20),
        ("Divya Menon", "divya.menon.rs@bmc.gov.in", "divyaRS@123", 21),
        ("Sameer Chatterjee", "sameer.chatterjee.rc@bmc.gov.in", "sameerRC@123", 22),
        ("Asha Reddy", "asha.reddy.s@bmc.gov.in", "ashaS@123", 23),
        ("Vivek Bansal", "vivek.bansal.t@bmc.gov.in", "vivekT@123", 24),
    ]

    cursor.executemany("""
        INSERT OR IGNORE INTO operators (name, email, password, ward_id)
        VALUES (?, ?, ?, ?)
    """, operators)

    conn.commit()
    conn.close()
    print("✅ Database initialized with all 24 ward operators.")

if __name__ == "__main__":
    init_db()
