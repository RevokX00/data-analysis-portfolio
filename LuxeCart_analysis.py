"""
LuxeCart E-Commerce Analysis
Author: Sadi Jagdish Reddy
Tools: Python, Pandas, NumPy, Matplotlib, Seaborn, SQL (via pandas)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings, os
warnings.filterwarnings('ignore')

os.makedirs('charts', exist_ok=True)

# ── LOAD DATA ────────────────────────────────────────────────────────────────
print("Loading data...")
orders   = pd.read_csv('data/orders.csv',      parse_dates=['order_date'])
items    = pd.read_csv('data/order_items.csv')
customers= pd.read_csv('data/customers.csv',   parse_dates=['registration_date'])
products = pd.read_csv('data/products.csv',    parse_dates=['launch_date'])
returns  = pd.read_csv('data/returns.csv',     parse_dates=['return_date'])

# ── CLEAN ────────────────────────────────────────────────────────────────────
print("Cleaning...")
orders['year']       = orders['order_date'].dt.year
orders['month']      = orders['order_date'].dt.month
orders['month_name'] = orders['order_date'].dt.strftime('%b')
orders['yearmonth']  = orders['order_date'].dt.to_period('M').astype(str)
orders['weekday']    = orders['order_date'].dt.day_name()
orders['hour']       = orders['order_date'].dt.hour
orders['quarter']    = orders['order_date'].dt.quarter

delivered = orders[orders['order_status'] == 'Delivered'].copy()

# ── STYLE ─────────────────────────────────────────────────────────────────────
BLUE  = '#1A56A0'; TEAL = '#2196A8'; RED = '#E74C3C'
GREEN = '#27AE60'; ORG  = '#E67E22'; PUR = '#8E44AD'
COLORS = [BLUE, TEAL, ORG, RED, GREEN, PUR, '#F39C12', '#1ABC9C']
plt.rcParams.update({'figure.facecolor':'white','axes.facecolor':'#F8F9FA',
                     'axes.spines.top':'False','axes.spines.right':'False',
                     'font.family':'sans-serif'})

def save(name):
    plt.tight_layout()
    plt.savefig(f'charts/{name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Chart saved: {name}")

# ════════════════════════════════════════════════════════════════════════════
# SECTION 1 — REVENUE PERFORMANCE
# ════════════════════════════════════════════════════════════════════════════
print("\n[1] Revenue Analysis...")

# 1.1 Monthly Revenue Trend
monthly_rev = orders.groupby('yearmonth')['total_amount'].sum().reset_index()
monthly_rev['yearmonth_dt'] = pd.to_datetime(monthly_rev['yearmonth'])
monthly_rev = monthly_rev.sort_values('yearmonth_dt')

fig, ax = plt.subplots(figsize=(14,5))
ax.fill_between(range(len(monthly_rev)), monthly_rev['total_amount'], alpha=0.15, color=BLUE)
ax.plot(range(len(monthly_rev)), monthly_rev['total_amount'], color=BLUE, linewidth=2.5, marker='o', markersize=3)
step = max(1, len(monthly_rev)//12)
ax.set_xticks(range(0, len(monthly_rev), step))
ax.set_xticklabels(monthly_rev['yearmonth'].iloc[::step], rotation=45, ha='right', fontsize=8)
ax.set_title('Monthly Revenue Trend — LuxeCart 2022–2024', fontsize=13, fontweight='bold', color=BLUE)
ax.set_ylabel('Revenue ($)'); ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'${x:,.0f}'))
save('01_monthly_revenue')

# 1.2 Annual Revenue Comparison
annual_rev = orders.groupby('year')['total_amount'].sum().reset_index()
fig, ax = plt.subplots(figsize=(8,5))
bars = ax.bar(annual_rev['year'].astype(str), annual_rev['total_amount'],
              color=[BLUE, TEAL, ORG], width=0.5, edgecolor='white')
for bar in bars:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2000,
            f'${bar.get_height():,.0f}', ha='center', fontweight='bold', fontsize=11)
ax.set_title('Annual Revenue — LuxeCart', fontsize=13, fontweight='bold', color=BLUE)
ax.set_ylabel('Total Revenue ($)')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'${x/1e6:.1f}M'))
save('02_annual_revenue')

# 1.3 Revenue by Country
rev_country = orders.groupby('country')['total_amount'].sum().sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(9,5))
bars = ax.barh(rev_country.index, rev_country.values, color=COLORS[:len(rev_country)])
for bar in bars:
    ax.text(bar.get_width()+1000, bar.get_y()+bar.get_height()/2,
            f'${bar.get_width():,.0f}', va='center', fontsize=10)
ax.set_title('Revenue by Country', fontsize=13, fontweight='bold', color=BLUE)
ax.set_xlabel('Total Revenue ($)')
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'${x/1e6:.1f}M'))
save('03_revenue_by_country')

# ════════════════════════════════════════════════════════════════════════════
# SECTION 2 — PRODUCT PERFORMANCE
# ════════════════════════════════════════════════════════════════════════════
print("\n[2] Product Analysis...")

# 2.1 Revenue by Category
merged = items.merge(orders[['order_id','total_amount','year','yearmonth','order_status']], on='order_id')
cat_rev = merged.groupby('category').agg(
    total_revenue=('unit_price','sum'),
    total_units=('quantity','sum'),
    avg_price=('unit_price','mean')
).reset_index().sort_values('total_revenue', ascending=False)

fig, axes = plt.subplots(1,2,figsize=(14,5))
axes[0].bar(cat_rev['category'], cat_rev['total_revenue'], color=COLORS[:len(cat_rev)], edgecolor='white')
axes[0].set_title('Revenue by Category', fontsize=12, fontweight='bold', color=BLUE)
axes[0].set_ylabel('Revenue ($)'); axes[0].tick_params(axis='x', rotation=30)
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'${x/1e6:.1f}M'))
axes[1].bar(cat_rev['category'], cat_rev['avg_price'], color=COLORS[:len(cat_rev)], edgecolor='white')
axes[1].set_title('Average Price by Category', fontsize=12, fontweight='bold', color=BLUE)
axes[1].set_ylabel('Avg Price ($)'); axes[1].tick_params(axis='x', rotation=30)
save('04_category_performance')

# 2.2 Top 10 Products by Revenue
prod_rev = merged.groupby(['product_id','product_name','category'])['unit_price'].sum().reset_index()
prod_rev = prod_rev.sort_values('unit_price', ascending=False).head(10)
fig, ax = plt.subplots(figsize=(12,6))
bars = ax.barh(prod_rev['product_name'][::-1], prod_rev['unit_price'][::-1], color=BLUE)
for bar in bars:
    ax.text(bar.get_width()+500, bar.get_y()+bar.get_height()/2,
            f'${bar.get_width():,.0f}', va='center', fontsize=9)
ax.set_title('Top 10 Products by Revenue', fontsize=13, fontweight='bold', color=BLUE)
ax.set_xlabel('Total Revenue ($)')
save('05_top_products')

# ════════════════════════════════════════════════════════════════════════════
# SECTION 3 — CUSTOMER BEHAVIOUR
# ════════════════════════════════════════════════════════════════════════════
print("\n[3] Customer Analysis...")

# 3.1 Acquisition Channel Performance
chan_rev = orders.groupby('acquisition_channel').agg(
    orders=('order_id','count'),
    revenue=('total_amount','sum'),
    avg_order=('total_amount','mean')
).reset_index().sort_values('revenue', ascending=False)

fig, axes = plt.subplots(1,2,figsize=(14,5))
axes[0].bar(chan_rev['acquisition_channel'], chan_rev['revenue'], color=COLORS[:len(chan_rev)], edgecolor='white')
axes[0].set_title('Revenue by Acquisition Channel', fontsize=12, fontweight='bold', color=BLUE)
axes[0].tick_params(axis='x', rotation=30); axes[0].set_ylabel('Revenue ($)')
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'${x/1e6:.1f}M'))
axes[1].bar(chan_rev['acquisition_channel'], chan_rev['avg_order'], color=COLORS[:len(chan_rev)], edgecolor='white')
axes[1].set_title('Average Order Value by Channel', fontsize=12, fontweight='bold', color=BLUE)
axes[1].tick_params(axis='x', rotation=30); axes[1].set_ylabel('Avg Order ($)')
save('06_acquisition_channels')

# 3.2 Repeat Purchase Analysis
order_counts = orders.groupby('customer_id')['order_id'].count().reset_index()
order_counts.columns = ['customer_id','order_count']
repeat_dist = order_counts['order_count'].value_counts().sort_index().head(10)
one_time = (order_counts['order_count']==1).sum()
repeat   = (order_counts['order_count']>1).sum()
repeat_rate = repeat / len(order_counts) * 100

fig, axes = plt.subplots(1,2,figsize=(14,5))
axes[0].bar(repeat_dist.index.astype(str), repeat_dist.values, color=BLUE, edgecolor='white')
axes[0].set_title('Orders Per Customer Distribution', fontsize=12, fontweight='bold', color=BLUE)
axes[0].set_xlabel('Number of Orders'); axes[0].set_ylabel('Number of Customers')
axes[1].pie([one_time, repeat], labels=[f'One-Time\n{one_time:,}', f'Repeat\n{repeat:,}'],
            autopct='%1.1f%%', colors=[TEAL, BLUE], startangle=90,
            wedgeprops={'edgecolor':'white','linewidth':2})
axes[1].set_title(f'Customer Loyalty — Repeat Rate: {repeat_rate:.1f}%', fontsize=12, fontweight='bold', color=BLUE)
save('07_repeat_purchase')

# 3.3 Age Group Analysis
age_rev = orders.merge(customers[['customer_id','age_group','gender']], on='customer_id')
age_stats = age_rev.groupby('age_group').agg(
    revenue=('total_amount','sum'),
    orders=('order_id','count'),
    aov=('total_amount','mean')
).reset_index()

fig, ax = plt.subplots(figsize=(10,5))
bars = ax.bar(age_stats['age_group'], age_stats['revenue'], color=COLORS[:len(age_stats)], edgecolor='white')
for bar in bars:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2000,
            f'${bar.get_height():,.0f}', ha='center', fontsize=9, fontweight='bold')
ax.set_title('Revenue by Age Group', fontsize=13, fontweight='bold', color=BLUE)
ax.set_ylabel('Total Revenue ($)')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'${x/1e6:.1f}M'))
save('08_age_group_revenue')

# ════════════════════════════════════════════════════════════════════════════
# SECTION 4 — SEASONALITY AND TRENDS
# ════════════════════════════════════════════════════════════════════════════
print("\n[4] Seasonality Analysis...")

# 4.1 Monthly Seasonality Heatmap
orders['month_num'] = orders['order_date'].dt.month
pivot = orders.groupby(['year','month_num'])['total_amount'].sum().unstack()
fig, ax = plt.subplots(figsize=(14,4))
sns.heatmap(pivot/1000, annot=True, fmt='.0f', cmap='Blues', ax=ax,
            xticklabels=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'],
            linewidths=0.5, cbar_kws={'label':'Revenue ($K)'})
ax.set_title('Revenue Heatmap by Month and Year ($K)', fontsize=13, fontweight='bold', color=BLUE)
ax.set_ylabel('Year')
save('09_seasonality_heatmap')

# 4.2 Day of Week Pattern
dow_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
dow_rev = orders.groupby('weekday')['total_amount'].mean().reindex(dow_order)
fig, ax = plt.subplots(figsize=(10,5))
bars = ax.bar(dow_rev.index, dow_rev.values, color=[RED if d in ['Saturday','Sunday'] else BLUE for d in dow_rev.index], edgecolor='white')
ax.set_title('Average Daily Revenue by Day of Week', fontsize=13, fontweight='bold', color=BLUE)
ax.set_ylabel('Avg Revenue ($)')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'${x:,.0f}'))
save('10_day_of_week')

# ════════════════════════════════════════════════════════════════════════════
# SECTION 5 — DISCOUNT ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
print("\n[5] Discount Analysis...")

disc_analysis = orders.groupby('discount_pct').agg(
    orders=('order_id','count'),
    revenue=('total_amount','sum'),
    aov=('total_amount','mean')
).reset_index()
disc_analysis['discount_label'] = disc_analysis['discount_pct'].apply(lambda x: f'{int(x*100)}% off' if x > 0 else 'No Discount')

fig, axes = plt.subplots(1,2,figsize=(14,5))
axes[0].bar(disc_analysis['discount_label'], disc_analysis['orders'], color=COLORS[:len(disc_analysis)], edgecolor='white')
axes[0].set_title('Orders by Discount Level', fontsize=12, fontweight='bold', color=BLUE)
axes[0].set_ylabel('Number of Orders'); axes[0].tick_params(axis='x', rotation=30)
axes[1].bar(disc_analysis['discount_label'], disc_analysis['aov'], color=COLORS[:len(disc_analysis)], edgecolor='white')
axes[1].set_title('Average Order Value by Discount Level', fontsize=12, fontweight='bold', color=BLUE)
axes[1].set_ylabel('Avg Order Value ($)'); axes[1].tick_params(axis='x', rotation=30)
save('11_discount_analysis')

# ════════════════════════════════════════════════════════════════════════════
# SECTION 6 — RETURNS ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
print("\n[6] Returns Analysis...")

ret_by_cat = returns.groupby('category').agg(
    returns=('return_id','count'),
    refund_value=('refund_amount','sum')
).reset_index()
total_by_cat = items.groupby('category')['quantity'].sum().reset_index()
total_by_cat.columns = ['category','total_sold']
ret_by_cat = ret_by_cat.merge(total_by_cat, on='category')
ret_by_cat['return_rate'] = ret_by_cat['returns'] / ret_by_cat['total_sold'] * 100

fig, axes = plt.subplots(1,2,figsize=(14,5))
colors_ret = [RED if r > 10 else ORG if r > 7 else GREEN for r in ret_by_cat['return_rate']]
axes[0].barh(ret_by_cat['category'], ret_by_cat['return_rate'], color=colors_ret)
axes[0].axvline(10, color='red', linestyle='--', linewidth=1, label='10% threshold')
axes[0].set_title('Return Rate by Category (%)', fontsize=12, fontweight='bold', color=BLUE)
axes[0].set_xlabel('Return Rate (%)')
axes[0].legend()

ret_reason = returns['return_reason'].value_counts()
axes[1].barh(ret_reason.index[::-1], ret_reason.values[::-1], color=BLUE)
axes[1].set_title('Return Reasons', fontsize=12, fontweight='bold', color=BLUE)
axes[1].set_xlabel('Number of Returns')
save('12_returns_analysis')

# ════════════════════════════════════════════════════════════════════════════
# SECTION 7 — PAYMENT METHODS
# ════════════════════════════════════════════════════════════════════════════
print("\n[7] Payment Analysis...")

pay_stats = orders.groupby('payment_method').agg(
    orders=('order_id','count'),
    revenue=('total_amount','sum'),
    aov=('total_amount','mean')
).reset_index().sort_values('revenue', ascending=False)

fig, axes = plt.subplots(1,2,figsize=(14,5))
axes[0].pie(pay_stats['revenue'], labels=pay_stats['payment_method'],
            autopct='%1.1f%%', colors=COLORS[:len(pay_stats)],
            startangle=90, wedgeprops={'edgecolor':'white','linewidth':2})
axes[0].set_title('Revenue Share by Payment Method', fontsize=12, fontweight='bold', color=BLUE)
axes[1].bar(pay_stats['payment_method'], pay_stats['aov'], color=COLORS[:len(pay_stats)], edgecolor='white')
axes[1].set_title('Average Order Value by Payment Method', fontsize=12, fontweight='bold', color=BLUE)
axes[1].set_ylabel('Avg Order ($)'); axes[1].tick_params(axis='x', rotation=20)
save('13_payment_methods')

# ════════════════════════════════════════════════════════════════════════════
# SECTION 8 — KPI SUMMARY
# ════════════════════════════════════════════════════════════════════════════
print("\n[8] Computing KPIs...")

total_revenue   = orders['total_amount'].sum()
total_orders    = len(orders)
aov             = orders['total_amount'].mean()
total_customers = customers['customer_id'].nunique()
repeat_rate_val = repeat_rate
total_returns   = len(returns)
return_rate_val = total_returns / len(items) * 100
total_discount  = orders['discount_amount'].sum()
disc_rate       = orders[orders['discount_pct']>0]['order_id'].count() / total_orders * 100

kpis = {
    'total_revenue':    round(total_revenue, 2),
    'total_orders':     total_orders,
    'aov':              round(aov, 2),
    'total_customers':  total_customers,
    'repeat_rate_pct':  round(repeat_rate_val, 1),
    'return_rate_pct':  round(return_rate_val, 1),
    'total_returns':    total_returns,
    'total_discounts':  round(total_discount, 2),
    'discounted_orders_pct': round(disc_rate, 1),
}

import json
with open('data/kpis.json','w') as f:
    json.dump(kpis, f)

print(f"\n{'='*50}")
print(f"LUXECART KEY PERFORMANCE INDICATORS")
print(f"{'='*50}")
for k,v in kpis.items():
    print(f"  {k:30s}: {v}")

print(f"\n✅ Analysis complete — {len(os.listdir('charts'))} charts generated")
