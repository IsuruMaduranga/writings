"""The task every experiment shares: 30 labelled tickets, the three teams, and how each method is asked."""

# 30 hand-labelled support tickets, 10 per team. A few are deliberately mixed.
TICKETS = [
 ("I was billed twice this month, please refund one of the charges.", "billing"),
 ("Can I get an invoice with my company's VAT number on it?", "billing"),
 ("My card was declined but my bank says there is money in the account.", "billing"),
 ("Why did my subscription price go up from 10 to 15 dollars?", "billing"),
 ("I cancelled last week but you still took a payment today.", "billing"),
 ("Do you offer a discount if I pay yearly instead of monthly?", "billing"),
 ("The receipt you emailed shows the wrong amount.", "billing"),
 ("Please switch my plan to the cheaper one from next month.", "billing"),
 ("I want a refund, the product did not work for me.", "billing"),
 ("Your payment page keeps timing out when I enter my card.", "billing"),
 ("The app crashes every time I open the settings screen.", "technical"),
 ("Export to CSV produces an empty file.", "technical"),
 ("Sync between my phone and laptop stopped working yesterday.", "technical"),
 ("I get a 500 error when I upload images larger than 5 MB.", "technical"),
 ("The dark mode makes the text unreadable on Android.", "technical"),
 ("Notifications arrive two hours late.", "technical"),
 ("Your API returns a timeout for every request since this morning.", "technical"),
 ("The search box does not find documents I created today.", "technical"),
 ("After the latest update the app uses all my battery.", "technical"),
 ("Charts on the dashboard show no data even though I have entries.", "technical"),
 ("I forgot my password and the reset email never arrives.", "account"),
 ("How do I change the email address on my profile?", "account"),
 ("Please delete my account and all my data.", "account"),
 ("Someone logged into my account from another country.", "account"),
 ("I want to add a second user to my team workspace.", "account"),
 ("How do I turn on two-factor authentication?", "account"),
 ("My username shows my old surname, can I change it?", "account"),
 ("I am locked out after too many login attempts.", "account"),
 ("Can I merge my two accounts into one?", "account"),
 ("Transfer ownership of the workspace to my colleague.", "account"),
]
LABELS = ["billing", "technical", "account"]

# Tickets that belong to no team, for testing what a model does when no answer fits.
NO_TEAM = ["What are your office hours?", "Do you have a job opening for a designer?",
           "asdf qwer zxcv", "I love your product, thank you!"]

# The prompt GPT-2 continues; the answer is its next token.
GPT2_PROMPT = ("A support ticket is routed to one team: billing, technical, or account.\n"
               "Ticket: {t}\nTeam:")

# The same question in Laya's (and Jev's) request format.
Q = {"team": {"type": "choice", "instructions": "Which team should handle this support ticket?",
              "criteria": {"billing": "payments, invoices, refunds, plans, prices",
                           "technical": "bugs, crashes, errors, things not working",
                           "account": "login, password, profile, users, account settings"}}}
