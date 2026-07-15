import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams['font.size'] = 11
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

orders = pd.read_csv('olist_orders.csv', parse_dates=['order_purchase_timestamp','order_delivered_date','order_estimated_delivery_date'])
items = pd.read_csv('olist_order_items.csv')
payments = pd.read_csv('olist_payments.csv')
customers = pd.read_csv('olist_customers.csv')

delivered = orders[orders['order_status']=='delivered'].copy()

# ---------- 1. Monthly revenue trend ----------
order_rev = items.groupby('order_id')['price'].sum().add(items.groupby('order_id')['freight_value'].sum(), fill_value=0).rename('order_value')
delivered = delivered.merge(order_rev, on='order_id', how='left')
delivered['month'] = delivered['order_purchase_timestamp'].dt.to_period('M')
monthly = delivered.groupby('month').agg(revenue=('order_value','sum'), orders=('order_id','count')).reset_index()
monthly['month_ts'] = monthly['month'].dt.to_timestamp()
monthly['mom_growth'] = monthly['revenue'].pct_change()*100

fig, ax1 = plt.subplots(figsize=(9,4.5))
ax1.bar(monthly['month_ts'], monthly['revenue'], width=20, color='#3B6E8F', label='Revenue')
ax1.set_ylabel('Monthly Revenue (R$)')
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,p: f'{x/1000:.0f}k'))
ax1.set_title('Monthly Revenue Trend (Oct 2016 – Sep 2018)')
ax1.set_xlabel('Month')
fig.tight_layout()
fig.savefig('chart_monthly_revenue.png', dpi=150)
plt.close(fig)

total_revenue = delivered['order_value'].sum()
avg_order_value = delivered['order_value'].mean()
first6 = monthly.head(6)['revenue'].sum()
last6 = monthly.tail(6)['revenue'].sum()
growth_h1_h2 = (last6-first6)/first6*100

# ---------- 2. Top categories ----------
items_delivered = items[items['order_id'].isin(delivered['order_id'])]
cat_rev = items_delivered.groupby('product_category').agg(revenue=('price','sum'), units=('price','count')).sort_values('revenue', ascending=False)
top10cat = cat_rev.head(10)

fig, ax = plt.subplots(figsize=(9,5))
ax.barh(top10cat.index[::-1], top10cat['revenue'][::-1], color='#3B6E8F')
ax.set_xlabel('Revenue (R$)')
ax.set_title('Top 10 Product Categories by Revenue')
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,p: f'{x/1000:.0f}k'))
fig.tight_layout()
fig.savefig('chart_top_categories.png', dpi=150)
plt.close(fig)

# ---------- 3. RFM segmentation ----------
snapshot_date = delivered['order_purchase_timestamp'].max() + pd.Timedelta(days=1)
rfm = delivered.groupby('customer_id').agg(
    recency=('order_purchase_timestamp', lambda x: (snapshot_date - x.max()).days),
    frequency=('order_id','count'),
    monetary=('order_value','sum')
).reset_index()

rfm['R_score'] = pd.qcut(rfm['recency'], 4, labels=[4,3,2,1]).astype(int)
rfm['F_score'] = rfm['frequency'].apply(lambda x: 4 if x>=3 else (3 if x==2 else (2 if x==1 else 1)))
rfm['M_score'] = pd.qcut(rfm['monetary'].rank(method='first'), 4, labels=[1,2,3,4]).astype(int)
rfm['RFM_sum'] = rfm['R_score']+rfm['F_score']+rfm['M_score']

def segment(row):
    if row['RFM_sum']>=10:
        return 'Champions'
    elif row['RFM_sum']>=8:
        return 'Loyal'
    elif row['RFM_sum']>=6:
        return 'Potential'
    elif row['recency']>180:
        return 'At Risk / Churned'
    else:
        return 'New / Low-Value'

rfm['segment'] = rfm.apply(segment, axis=1)
seg_counts = rfm['segment'].value_counts()
seg_revenue = rfm.groupby('segment')['monetary'].sum().sort_values(ascending=False)

fig, axes = plt.subplots(1,2, figsize=(11,4.5))
colors = ['#3B6E8F','#6FA3C7','#A8CBE0','#E0A458','#C65F4A']
axes[0].pie(seg_counts, labels=seg_counts.index, autopct='%1.0f%%', colors=colors, startangle=90, textprops={'fontsize':9})
axes[0].set_title('Customer Segments by Count')
axes[1].barh(seg_revenue.index[::-1], seg_revenue.values[::-1], color='#3B6E8F')
axes[1].set_title('Revenue Contribution by Segment')
axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,p: f'{x/1000:.0f}k'))
fig.tight_layout()
fig.savefig('chart_rfm_segments.png', dpi=150)
plt.close(fig)

repeat_rate = (rfm['frequency']>1).mean()*100

# ---------- 4. Delivery time vs review score ----------
delivered_valid = delivered.dropna(subset=['delivery_delta','review_score'])
bins = [-100,-10,-3,0,3,10,100]
labels = ['10+ days early','3-10 days early','0-3 days early','0-3 days late','3-10 days late','10+ days late']
delivered_valid['delay_bucket'] = pd.cut(delivered_valid['delivery_delta'], bins=bins, labels=labels)
delay_review = delivered_valid.groupby('delay_bucket')['review_score'].mean().reindex(labels)

fig, ax = plt.subplots(figsize=(9,4.5))
bar_colors = ['#3B6E8F']*3 + ['#E0A458','#D9834E','#C65F4A']
ax.bar(delay_review.index, delay_review.values, color=bar_colors)
ax.set_ylabel('Average Review Score (1-5)')
ax.set_title('Delivery Timing vs. Average Review Score')
ax.set_ylim(1,5)
plt.setp(ax.get_xticklabels(), rotation=20, ha='right')
fig.tight_layout()
fig.savefig('chart_delivery_vs_review.png', dpi=150)
plt.close(fig)

corr = delivered_valid['delivery_delta'].corr(delivered_valid['review_score'])
late_pct = (delivered_valid['delivery_delta']>0).mean()*100
avg_review_ontime = delivered_valid[delivered_valid['delivery_delta']<=0]['review_score'].mean()
avg_review_late = delivered_valid[delivered_valid['delivery_delta']>0]['review_score'].mean()

# ---------- 5. Geographic breakdown ----------
state_rev = delivered.groupby('customer_state')['order_value'].sum().sort_values(ascending=False)
top8states = state_rev.head(8)

fig, ax = plt.subplots(figsize=(9,4.5))
ax.bar(top8states.index, top8states.values, color='#3B6E8F')
ax.set_ylabel('Revenue (R$)')
ax.set_title('Revenue by Top 8 Customer States')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,p: f'{x/1000:.0f}k'))
fig.tight_layout()
fig.savefig('chart_state_revenue.png', dpi=150)
plt.close(fig)

state_review = delivered.groupby('customer_state')['review_score'].mean().sort_values()
worst_state_delay = orders.groupby('customer_state')['delivery_days'].mean().sort_values(ascending=False).head(3)

# ---------- Print summary stats for report writing ----------
print("=== KEY METRICS ===")
print(f"Total delivered orders: {len(delivered):,}")
print(f"Total revenue: R$ {total_revenue:,.2f}")
print(f"Average order value: R$ {avg_order_value:,.2f}")
print(f"Revenue growth (first 6mo vs last 6mo): {growth_h1_h2:.1f}%")
print()
print("Top 5 categories by revenue:")
print(cat_rev.head(5))
print()
print("RFM segment counts:")
print(seg_counts)
print()
print("RFM segment revenue:")
print(seg_revenue)
print(f"Repeat purchase rate: {repeat_rate:.1f}%")
print()
print(f"Correlation delivery_delta vs review_score: {corr:.3f}")
print(f"% of orders delivered late: {late_pct:.1f}%")
print(f"Avg review (on-time/early): {avg_review_ontime:.2f}")
print(f"Avg review (late): {avg_review_late:.2f}")
print()
print("Top states by revenue:")
print(top8states)
print()
print("Worst average review states (bottom 5):")
print(state_review.head(5))
print()
print("States with highest avg delivery days:")
print(worst_state_delay)

rfm.to_csv('rfm_output.csv', index=False)
