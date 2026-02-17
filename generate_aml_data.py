import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from faker import Faker
import sqlite3
import random
import os

fake = Faker('en_US')
np.random.seed(42)
random.seed(42)

# Configuration
NUM_CUSTOMERS = 1000
NUM_SUSPICIOUS = 100  # 10% of customers engage in suspicious activity
SIMULATION_DAYS = 365
START_DATE = datetime(2025, 1, 1)

# Money laundering typologies (10 types)
TYPOLOGIES = [
    'structuring', 'rapid_movement', 'layering', 'trade_based',
    'cash_intensive', 'shell_company', 'funnel_account',
    'third_party_payments', 'round_tripping', 'smurfing'
]

def generate_customers(num_customers, num_suspicious):
    """Generate customer profiles including suspicious actors"""
    print(f"Generating {num_customers:,} customer profiles...")

    customers = []
    suspicious_indices = np.random.choice(num_customers, num_suspicious, replace=False)

    for i in range(num_customers):
        is_suspicious = i in suspicious_indices

        # Customer type
        customer_type = np.random.choice(['individual', 'business'], p=[0.7, 0.3])

        if customer_type == 'individual':
            name = fake.name()
            occupation = fake.job() if not is_suspicious else np.random.choice([
                'Self-Employed', 'Consultant', 'Freelancer', 'Unemployed'
            ])
        else:
            name = fake.company()
            occupation = fake.bs()

        # Income/revenue
        if is_suspicious and customer_type == 'business':
            # Suspicious businesses often have low stated income but high transactions
            annual_income = np.random.uniform(50000, 200000)
        else:
            annual_income = np.random.lognormal(np.log(75000), 0.8)
            annual_income = np.clip(annual_income, 20000, 5000000)

        # Assign typology to suspicious customers
        if is_suspicious:
            typology = np.random.choice(TYPOLOGIES)
        else:
            typology = None

        customers.append({
            'customer_id': i,
            'name': name,
            'customer_type': customer_type,
            'occupation': occupation,
            'annual_income': round(annual_income, 2),
            'address': fake.address().replace('\n', ', '),
            'city': fake.city(),
            'state': fake.state_abbr(),
            'zip_code': fake.zipcode(),
            'ssn_ein': fake.ssn() if customer_type == 'individual' else fake.ein(),
            'account_open_date': (START_DATE - timedelta(days=np.random.randint(30, 3650))).strftime('%Y-%m-%d'),
            'risk_rating': 'high' if is_suspicious else np.random.choice(['low', 'medium'], p=[0.7, 0.3]),
            'is_suspicious': is_suspicious,
            'typology': typology
        })

    return pd.DataFrame(customers)

def generate_structuring_pattern(customer, start_date, num_days=30):
    """Generate structuring pattern: multiple deposits just under $10K"""
    transactions = []

    # 15-25 deposits over the period
    num_deposits = np.random.randint(15, 26)

    for i in range(num_deposits):
        # Amount just under $10K threshold
        amount = np.random.uniform(7000, 9900)

        # Random day within period
        days_offset = np.random.randint(0, num_days)
        trans_date = start_date + timedelta(days=days_offset)

        # Alternate between different branches
        branch = np.random.choice([
            'Downtown Branch', 'Westside Branch', 'Airport Branch',
            'Suburban Branch', 'Mall Branch'
        ])

        transactions.append({
            'customer_id': customer['customer_id'],
            'transaction_date': trans_date.strftime('%Y-%m-%d %H:%M:%S'),
            'transaction_type': 'deposit',
            'amount': round(amount, 2),
            'method': 'cash',
            'location': branch,
            'description': 'Cash Deposit',
            'counterparty': None,
            'counterparty_account': None,
            'counterparty_bank': None,
            'country': 'USA'
        })

    return transactions

def generate_rapid_movement_pattern(customer, start_date, num_cycles=5):
    """Generate rapid movement: deposit -> wire out -> withdrawal"""
    transactions = []

    for cycle in range(num_cycles):
        cycle_start = start_date + timedelta(days=cycle * 30)

        # Large deposit
        deposit_amount = np.random.uniform(25000, 150000)

        transactions.append({
            'customer_id': customer['customer_id'],
            'transaction_date': cycle_start.strftime('%Y-%m-%d %H:%M:%S'),
            'transaction_type': 'deposit',
            'amount': round(deposit_amount, 2),
            'method': 'wire',
            'location': 'Wire Transfer',
            'description': 'Incoming Wire',
            'counterparty': fake.company(),
            'counterparty_account': fake.bban(),
            'counterparty_bank': fake.company() + ' Bank',
            'country': np.random.choice(['China', 'Russia', 'UAE', 'Panama', 'Cyprus'])
        })

        # Immediate wire transfer out (within 24-48 hours)
        wire_date = cycle_start + timedelta(hours=np.random.randint(24, 72))
        wire_amount = deposit_amount * np.random.uniform(0.95, 0.99)  # Keep small amount

        transactions.append({
            'customer_id': customer['customer_id'],
            'transaction_date': wire_date.strftime('%Y-%m-%d %H:%M:%S'),
            'transaction_type': 'withdrawal',
            'amount': round(wire_amount, 2),
            'method': 'wire',
            'location': 'Wire Transfer',
            'description': 'Outgoing Wire',
            'counterparty': fake.name(),
            'counterparty_account': fake.bban(),
            'counterparty_bank': fake.company() + ' Bank',
            'country': np.random.choice(['Cayman Islands', 'Switzerland', 'Singapore', 'Hong Kong'])
        })

    return transactions

def generate_cash_intensive_pattern(customer, start_date, num_months=12):
    """Generate suspicious cash-intensive business deposits"""
    transactions = []

    # Industry benchmark (what SHOULD be deposited monthly)
    industry_benchmark = customer['annual_income'] / 12 * 0.6  # 60% cash business

    # Suspicious: 3-5x the industry benchmark
    monthly_cash = industry_benchmark * np.random.uniform(3, 5)

    for month in range(num_months):
        # 8-12 deposits per month
        num_deposits = np.random.randint(8, 13)

        for dep in range(num_deposits):
            amount = monthly_cash / num_deposits * np.random.uniform(0.8, 1.2)

            deposit_date = start_date + timedelta(days=month*30 + np.random.randint(0, 30))

            transactions.append({
                'customer_id': customer['customer_id'],
                'transaction_date': deposit_date.strftime('%Y-%m-%d %H:%M:%S'),
                'transaction_type': 'deposit',
                'amount': round(amount, 2),
                'method': 'cash',
                'location': 'Main Branch',
                'description': f'Business Cash Deposit - {customer["name"]}',
                'counterparty': None,
                'counterparty_account': None,
                'counterparty_bank': None,
                'country': 'USA'
            })

    return transactions

def generate_shell_company_pattern(customer, start_date, num_wires=20):
    """Generate shell company pattern: high-value wires with no business activity"""
    transactions = []

    for i in range(num_wires):
        days_offset = np.random.randint(0, 365)
        trans_date = start_date + timedelta(days=days_offset)

        # Incoming wire
        amount_in = np.random.uniform(100000, 500000)
        transactions.append({
            'customer_id': customer['customer_id'],
            'transaction_date': trans_date.strftime('%Y-%m-%d %H:%M:%S'),
            'transaction_type': 'deposit',
            'amount': round(amount_in, 2),
            'method': 'wire',
            'location': 'Wire Transfer',
            'description': 'International Wire - Consulting Fees',
            'counterparty': fake.company(),
            'counterparty_account': fake.bban(),
            'counterparty_bank': fake.company() + ' Bank',
            'country': np.random.choice(['British Virgin Islands', 'Seychelles', 'Belize', 'Malta'])
        })

        # Outgoing wire (2-5 days later)
        wire_out_date = trans_date + timedelta(days=np.random.randint(2, 6))
        amount_out = amount_in * np.random.uniform(0.92, 0.98)

        transactions.append({
            'customer_id': customer['customer_id'],
            'transaction_date': wire_out_date.strftime('%Y-%m-%d %H:%M:%S'),
            'transaction_type': 'withdrawal',
            'amount': round(amount_out, 2),
            'method': 'wire',
            'location': 'Wire Transfer',
            'description': 'Wire Transfer - Vendor Payment',
            'counterparty': fake.company(),
            'counterparty_account': fake.bban(),
            'counterparty_bank': fake.company() + ' Bank',
            'country': np.random.choice(['Luxembourg', 'Panama', 'Cyprus', 'Mauritius'])
        })

    return transactions

def generate_smurfing_pattern(customer, start_date, num_events=10):
    """Generate smurfing: coordinated deposits by multiple people"""
    transactions = []

    # Network of 3-5 "smurfs" (money mules)
    num_smurfs = np.random.randint(3, 6)
    smurf_names = [fake.name() for _ in range(num_smurfs)]

    for event in range(num_events):
        event_date = start_date + timedelta(days=event * 15)

        # Each smurf makes a deposit on the same day or within 24 hours
        for smurf_name in smurf_names:
            amount = np.random.uniform(3000, 9000)
            deposit_time = event_date + timedelta(hours=np.random.randint(0, 24))

            branch = np.random.choice([
                'North Branch', 'South Branch', 'East Branch',
                'West Branch', 'Central Branch', 'Airport Branch'
            ])

            transactions.append({
                'customer_id': customer['customer_id'],
                'transaction_date': deposit_time.strftime('%Y-%m-%d %H:%M:%S'),
                'transaction_type': 'deposit',
                'amount': round(amount, 2),
                'method': 'cash',
                'location': branch,
                'description': f'Cash Deposit by {smurf_name}',
                'counterparty': smurf_name,
                'counterparty_account': None,
                'counterparty_bank': None,
                'country': 'USA'
            })

    return transactions

def generate_third_party_pattern(customer, start_date, num_payments=15):
    """Generate suspicious third-party payments"""
    transactions = []

    # Generate list of third parties
    third_parties = [fake.name() for _ in range(8)]

    for i in range(num_payments):
        payment_date = start_date + timedelta(days=np.random.randint(0, 365))

        # Receive from third party
        receive_amount = np.random.uniform(5000, 25000)
        transactions.append({
            'customer_id': customer['customer_id'],
            'transaction_date': payment_date.strftime('%Y-%m-%d %H:%M:%S'),
            'transaction_type': 'deposit',
            'amount': round(receive_amount, 2),
            'method': 'wire',
            'location': 'Wire Transfer',
            'description': 'Third Party Transfer',
            'counterparty': np.random.choice(third_parties),
            'counterparty_account': fake.bban(),
            'counterparty_bank': fake.company() + ' Bank',
            'country': 'USA'
        })

        # Pay to different third party (1-3 days later)
        pay_date = payment_date + timedelta(days=np.random.randint(1, 4))
        pay_amount = receive_amount * np.random.uniform(0.85, 0.95)

        transactions.append({
            'customer_id': customer['customer_id'],
            'transaction_date': pay_date.strftime('%Y-%m-%d %H:%M:%S'),
            'transaction_type': 'withdrawal',
            'amount': round(pay_amount, 2),
            'method': 'wire',
            'location': 'Wire Transfer',
            'description': 'Payment to Third Party',
            'counterparty': np.random.choice(third_parties),
            'counterparty_account': fake.bban(),
            'counterparty_bank': fake.company() + ' Bank',
            'country': 'USA'
        })

    return transactions

def generate_layering_pattern(customer, start_date, num_chains=5):
    """Generate layering: complex transfer chains"""
    transactions = []

    for chain in range(num_chains):
        chain_start = start_date + timedelta(days=chain * 60)

        # Initial large deposit
        initial_amount = np.random.uniform(50000, 200000)
        transactions.append({
            'customer_id': customer['customer_id'],
            'transaction_date': chain_start.strftime('%Y-%m-%d %H:%M:%S'),
            'transaction_type': 'deposit',
            'amount': round(initial_amount, 2),
            'method': 'wire',
            'location': 'Wire Transfer',
            'description': 'Wire Transfer In',
            'counterparty': fake.company(),
            'counterparty_account': fake.bban(),
            'counterparty_bank': fake.company() + ' Bank',
            'country': 'USA'
        })

        # Series of transfers (5-8 hops)
        num_hops = np.random.randint(5, 9)
        current_amount = initial_amount
        current_date = chain_start

        for hop in range(num_hops):
            current_date += timedelta(days=np.random.randint(3, 10))
            current_amount *= np.random.uniform(0.95, 0.99)

            # Alternate deposit/withdrawal
            trans_type = 'withdrawal' if hop % 2 == 1 else 'deposit'

            transactions.append({
                'customer_id': customer['customer_id'],
                'transaction_date': current_date.strftime('%Y-%m-%d %H:%M:%S'),
                'transaction_type': trans_type,
                'amount': round(current_amount, 2),
                'method': 'wire',
                'location': 'Wire Transfer',
                'description': f'Layering Transfer {hop+1}',
                'counterparty': fake.company(),
                'counterparty_account': fake.bban(),
                'counterparty_bank': fake.company() + ' Bank',
                'country': np.random.choice(['USA', 'UK', 'Germany', 'Singapore', 'Switzerland'])
            })

    return transactions

def generate_round_tripping_pattern(customer, start_date, num_cycles=4):
    """Generate round-tripping: funds leave and return via complex route"""
    transactions = []

    for cycle in range(num_cycles):
        cycle_start = start_date + timedelta(days=cycle * 90)
        amount = np.random.uniform(50000, 300000)

        # Step 1: Large outgoing wire
        transactions.append({
            'customer_id': customer['customer_id'],
            'transaction_date': cycle_start.strftime('%Y-%m-%d %H:%M:%S'),
            'transaction_type': 'withdrawal',
            'amount': round(amount, 2),
            'method': 'wire',
            'location': 'Wire Transfer',
            'description': 'Investment Wire - Overseas',
            'counterparty': fake.company(),
            'counterparty_account': fake.bban(),
            'counterparty_bank': fake.company() + ' Bank',
            'country': np.random.choice(['Cayman Islands', 'British Virgin Islands', 'Panama'])
        })

        # Step 2: Money returns as "investment income" (30-60 days later)
        return_date = cycle_start + timedelta(days=np.random.randint(30, 60))
        return_amount = amount * np.random.uniform(1.02, 1.10)  # Slight "profit"

        transactions.append({
            'customer_id': customer['customer_id'],
            'transaction_date': return_date.strftime('%Y-%m-%d %H:%M:%S'),
            'transaction_type': 'deposit',
            'amount': round(return_amount, 2),
            'method': 'wire',
            'location': 'Wire Transfer',
            'description': 'Investment Return - Foreign Entity',
            'counterparty': fake.company() + ' Holdings Ltd',
            'counterparty_account': fake.bban(),
            'counterparty_bank': fake.company() + ' Bank',
            'country': np.random.choice(['Luxembourg', 'Switzerland', 'Singapore'])
        })

    return transactions

def generate_funnel_account_pattern(customer, start_date, num_cycles=8):
    """Generate funnel account: many sources -> one account -> distribution"""
    transactions = []

    # Multiple source names
    sources = [fake.name() for _ in range(6)]
    # Few beneficiaries
    beneficiaries = [fake.name() for _ in range(2)]

    for cycle in range(num_cycles):
        cycle_start = start_date + timedelta(days=cycle * 45)

        # Collection phase: multiple incoming transfers
        total_collected = 0
        for source in sources:
            amount = np.random.uniform(5000, 20000)
            total_collected += amount
            deposit_date = cycle_start + timedelta(days=np.random.randint(0, 5))

            transactions.append({
                'customer_id': customer['customer_id'],
                'transaction_date': deposit_date.strftime('%Y-%m-%d %H:%M:%S'),
                'transaction_type': 'deposit',
                'amount': round(amount, 2),
                'method': 'wire',
                'location': 'Wire Transfer',
                'description': 'Transfer from Associate',
                'counterparty': source,
                'counterparty_account': fake.bban(),
                'counterparty_bank': fake.company() + ' Bank',
                'country': 'USA'
            })

        # Distribution phase: send to beneficiaries (5-7 days later)
        remaining = total_collected * np.random.uniform(0.90, 0.95)
        for beneficiary in beneficiaries:
            send_amount = remaining / len(beneficiaries) * np.random.uniform(0.9, 1.1)
            send_date = cycle_start + timedelta(days=np.random.randint(5, 8))

            transactions.append({
                'customer_id': customer['customer_id'],
                'transaction_date': send_date.strftime('%Y-%m-%d %H:%M:%S'),
                'transaction_type': 'withdrawal',
                'amount': round(send_amount, 2),
                'method': 'wire',
                'location': 'Wire Transfer',
                'description': 'Distribution Payment',
                'counterparty': beneficiary,
                'counterparty_account': fake.bban(),
                'counterparty_bank': fake.company() + ' Bank',
                'country': 'USA'
            })

    return transactions

def generate_trade_based_pattern(customer, start_date, num_invoices=8):
    """Generate trade-based money laundering: over-invoiced trade"""
    transactions = []

    trade_goods = ['Electronics', 'Textiles', 'Auto Parts', 'Machinery', 'Pharmaceuticals']

    for i in range(num_invoices):
        invoice_date = start_date + timedelta(days=np.random.randint(0, 365))

        # Over-invoiced amount (2-5x fair market value)
        fair_value = np.random.uniform(10000, 50000)
        invoiced_amount = fair_value * np.random.uniform(2, 5)

        # Incoming payment for "goods"
        transactions.append({
            'customer_id': customer['customer_id'],
            'transaction_date': invoice_date.strftime('%Y-%m-%d %H:%M:%S'),
            'transaction_type': 'deposit',
            'amount': round(invoiced_amount, 2),
            'method': 'wire',
            'location': 'Wire Transfer',
            'description': f'Trade Payment - {np.random.choice(trade_goods)}',
            'counterparty': fake.company(),
            'counterparty_account': fake.bban(),
            'counterparty_bank': fake.company() + ' Bank',
            'country': np.random.choice(['China', 'Turkey', 'India', 'Nigeria', 'Vietnam'])
        })

        # Outgoing "supplier payment" (much less than received)
        supplier_date = invoice_date + timedelta(days=np.random.randint(5, 15))
        supplier_amount = fair_value * np.random.uniform(0.8, 1.0)  # Near fair value

        transactions.append({
            'customer_id': customer['customer_id'],
            'transaction_date': supplier_date.strftime('%Y-%m-%d %H:%M:%S'),
            'transaction_type': 'withdrawal',
            'amount': round(supplier_amount, 2),
            'method': 'wire',
            'location': 'Wire Transfer',
            'description': f'Supplier Payment - {np.random.choice(trade_goods)}',
            'counterparty': fake.company(),
            'counterparty_account': fake.bban(),
            'counterparty_bank': fake.company() + ' Bank',
            'country': np.random.choice(['China', 'Turkey', 'India'])
        })

    return transactions

def generate_normal_transactions(customer, start_date, num_days):
    """Generate normal banking activity for non-suspicious customers"""
    transactions = []

    # Monthly income deposit
    monthly_income = customer['annual_income'] / 12

    for month in range(12):
        # Salary deposit
        salary_date = start_date + timedelta(days=month*30 + 1)
        transactions.append({
            'customer_id': customer['customer_id'],
            'transaction_date': salary_date.strftime('%Y-%m-%d %H:%M:%S'),
            'transaction_type': 'deposit',
            'amount': round(monthly_income * np.random.uniform(0.95, 1.05), 2),
            'method': 'ach' if customer['customer_type'] == 'individual' else 'wire',
            'location': 'ACH Deposit',
            'description': 'Salary' if customer['customer_type'] == 'individual' else 'Revenue',
            'counterparty': fake.company(),
            'counterparty_account': None,
            'counterparty_bank': None,
            'country': 'USA'
        })

        # Random expenses (5-15 per month)
        num_expenses = np.random.randint(5, 16)
        for _ in range(num_expenses):
            expense_date = start_date + timedelta(days=month*30 + np.random.randint(0, 30))
            amount = np.random.uniform(50, 2000)

            transactions.append({
                'customer_id': customer['customer_id'],
                'transaction_date': expense_date.strftime('%Y-%m-%d %H:%M:%S'),
                'transaction_type': 'withdrawal',
                'amount': round(amount, 2),
                'method': np.random.choice(['card', 'ach', 'check']),
                'location': fake.company(),
                'description': np.random.choice(['Purchase', 'Bill Payment', 'ATM Withdrawal']),
                'counterparty': fake.company(),
                'counterparty_account': None,
                'counterparty_bank': None,
                'country': 'USA'
            })

    return transactions

def generate_all_transactions(customers_df, start_date):
    """Generate all transactions based on customer typology"""
    print("\nGenerating transactions for all customers...")

    all_transactions = []

    for idx, customer in customers_df.iterrows():
        if customer['is_suspicious']:
            # Generate suspicious pattern based on typology
            typology = customer['typology']

            if typology == 'structuring':
                txns = generate_structuring_pattern(customer, start_date)
            elif typology == 'rapid_movement':
                txns = generate_rapid_movement_pattern(customer, start_date)
            elif typology == 'cash_intensive':
                txns = generate_cash_intensive_pattern(customer, start_date)
            elif typology == 'shell_company':
                txns = generate_shell_company_pattern(customer, start_date)
            elif typology == 'smurfing':
                txns = generate_smurfing_pattern(customer, start_date)
            elif typology == 'third_party_payments':
                txns = generate_third_party_pattern(customer, start_date)
            elif typology == 'layering':
                txns = generate_layering_pattern(customer, start_date)
            elif typology == 'round_tripping':
                txns = generate_round_tripping_pattern(customer, start_date)
            elif typology == 'funnel_account':
                txns = generate_funnel_account_pattern(customer, start_date)
            elif typology == 'trade_based':
                txns = generate_trade_based_pattern(customer, start_date)
            else:
                # Default to structuring for any unmapped typology
                txns = generate_structuring_pattern(customer, start_date)

            all_transactions.extend(txns)

            # Also add some normal transactions to mix
            normal_txns = generate_normal_transactions(customer, start_date, 365)
            all_transactions.extend(normal_txns[:20])  # Add 20 normal transactions
        else:
            # Normal customer
            txns = generate_normal_transactions(customer, start_date, 365)
            all_transactions.extend(txns)

        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx+1:,} / {len(customers_df):,} customers")

    return pd.DataFrame(all_transactions)

def create_alerts(customers_df, transactions_df):
    """Create alert records for suspicious customers"""
    print("\nGenerating alerts for suspicious activity...")

    alerts = []
    alert_id = 0

    suspicious_customers = customers_df[customers_df['is_suspicious']]

    for _, customer in suspicious_customers.iterrows():
        customer_txns = transactions_df[transactions_df['customer_id'] == customer['customer_id']]

        # Alert details based on typology
        typology = customer['typology']

        # Calculate key metrics
        total_volume = customer_txns['amount'].sum()
        num_transactions = len(customer_txns)
        avg_transaction = customer_txns['amount'].mean()

        # Create alert
        alerts.append({
            'alert_id': alert_id,
            'customer_id': customer['customer_id'],
            'alert_date': customer_txns['transaction_date'].max(),
            'alert_type': typology,
            'severity': 'high',
            'total_amount': round(total_volume, 2),
            'num_transactions': num_transactions,
            'avg_transaction_amount': round(avg_transaction, 2),
            'status': 'open',
            'assigned_analyst': fake.name(),
            'sar_filed': False
        })

        alert_id += 1

    return pd.DataFrame(alerts)

def save_to_database(customers_df, transactions_df, alerts_df, db_path='aml_data.db'):
    """Save all data to SQLite database"""
    print(f"\nSaving to database: {db_path}")

    conn = sqlite3.connect(db_path)

    customers_df.to_sql('customers', conn, if_exists='replace', index=False)
    transactions_df.to_sql('transactions', conn, if_exists='replace', index=False)
    alerts_df.to_sql('alerts', conn, if_exists='replace', index=False)

    conn.close()
    print(f"[OK] Database created: {db_path}")
    print(f"  - Customers: {len(customers_df):,} rows")
    print(f"  - Transactions: {len(transactions_df):,} rows")
    print(f"  - Alerts: {len(alerts_df):,} rows")

# MAIN EXECUTION
if __name__ == "__main__":
    print("=" * 60)
    print("AML SYNTHETIC DATA GENERATION")
    print("=" * 60)

    # Generate customers
    customers_df = generate_customers(NUM_CUSTOMERS, NUM_SUSPICIOUS)

    # Generate transactions
    transactions_df = generate_all_transactions(customers_df, START_DATE)

    # Add transaction IDs
    transactions_df['transaction_id'] = range(len(transactions_df))
    transactions_df = transactions_df.sort_values(['customer_id', 'transaction_date'])

    # Create alerts
    alerts_df = create_alerts(customers_df, transactions_df)

    # Save to database
    save_to_database(customers_df, transactions_df, alerts_df)

    # Save CSVs
    os.makedirs('data/generated', exist_ok=True)
    customers_df.to_csv('data/generated/customers.csv', index=False)
    transactions_df.to_csv('data/generated/transactions.csv', index=False)
    alerts_df.to_csv('data/generated/alerts.csv', index=False)

    print("\n" + "=" * 60)
    print("DATA GENERATION COMPLETE!")
    print("=" * 60)
    print(f"\nStatistics:")
    print(f"  Total customers: {len(customers_df):,}")
    print(f"  Suspicious customers: {customers_df['is_suspicious'].sum():,}")
    print(f"  Total transactions: {len(transactions_df):,}")
    print(f"  Total alerts: {len(alerts_df):,}")
    print(f"\nTypology Distribution:")
    print(customers_df[customers_df['is_suspicious']]['typology'].value_counts().to_string())
