import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import logging



# Promotion Distribution over train and test datasets 
def promo_distr(train, test):

    '''
        Fuction for checking the distribution of promotion in train and test datas.
    '''
    # Calculate promotion distribution
    train_promo_distribution = train['Promo'].value_counts(normalize=True) * 100
    test_promo_distribution = test['Promo'].value_counts(normalize=True) * 100

    # Combine distributions into a single DataFrame for easier plotting
    promo_comparison = pd.DataFrame({
        'Train': train_promo_distribution,
        'Test': test_promo_distribution
    }) 

    # Display the promotion comparison
    print(promo_comparison)

    # Plotting the distribution
    promo_comparison.plot(kind='bar', figsize=(8, 5))
    plt.title('Promotion Distribution: Training vs Test Set')
    plt.ylabel('Percentage (%)')
    plt.xticks(rotation=0)
    plt.legend(title='Dataset')
    plt.show()


# Sales behavior around holidat 
def get_sales_behavior(df, holiday_dates, days_before=1, days_after=1):
    '''
        Sales behavior around holiday 
        returns the sales before, during and after holiday in a day gap
        the day before and after can be changed to the desired number of date 
    '''
    sales_behavior = {}
    
    for holiday in holiday_dates:
        start_before = holiday - pd.Timedelta(days=days_before)
        end_before = holiday - pd.Timedelta(days=1)
        start_after = holiday + pd.Timedelta(days=1)
        end_after = holiday + pd.Timedelta(days=days_after)
        
        sales_behavior[holiday] = {
            'Before': df.loc[start_before:end_before]['Sales'].sum(),
            'During': df.loc[holiday]['Sales'].sum(),
            'After': df.loc[start_after:end_after]['Sales'].sum()
        }
    
    return sales_behavior


def salesholiday(df):
    '''
        Weekly Sales Over time with holidays highlights
    '''
    # Aggregate sales by week
    df['Week'] = df.index.to_period('W')
    weekly_sales = df.groupby('Week')['Sales'].sum().reset_index()

    # Plot sales with highlights for holidays
    plt.figure(figsize=(12, 6))
    plt.plot(weekly_sales['Week'].dt.start_time, weekly_sales['Sales'], label='Total Sales', color='blue')

    # public holiday
    publicholidays = df[df['StateHoliday']== 'a']
    plt.scatter(publicholidays.index, publicholidays['Sales'], color='green', label='public Holiday Sales', alpha=0.25)

    # Easter
    easterholidays = df[df['StateHoliday']== 'b']
    plt.scatter(easterholidays.index, easterholidays['Sales'], color='red', label='Easter Holiday Sales', alpha=0.5)

    # Christmas
    christmasholidays = df[df['StateHoliday']== 'c']
    plt.scatter(christmasholidays.index, christmasholidays['Sales'], color='skyblue', label='Christmas Holiday Sales', alpha=0.5)

    plt.title('Weekly Sales Over Time with Holiday Highlights')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.grid(True)
    plt.legend()
    plt.show()


def weeklymaxmin(df):
    '''
        shows the maximum and minimum of weeky sales
    '''
    # Resample by week and sum the sales
    weekly_sale= df.resample('W')['Sales'].sum()

    # Find the week with the highest sales
    highest_week = weekly_sale.idxmax()
    highest_sales_value = weekly_sale.max()

    # Find the week with the highest sales
    lowest_week = weekly_sale.idxmin()
    lowest_sales_value = weekly_sale.min()

    # Display the results
    print(f"The week with the highest sales is: {highest_week.strftime('%Y-%m-%d')}")
    print(f"Total sales for that week: {highest_sales_value}\n")

    print(f"The week with the lowest sales is: {lowest_week.strftime('%Y-%m-%d')}")
    print(f"Total sales for that week: {lowest_sales_value}")


def salesbyholiday(df):
    '''
        plot to see sales by holiday
    '''
    # Mapping StateHoliday values to more meaningful categories
    holiday_map = {
        'a': 'Public Holiday',
        'b': 'Easter',
        'c': 'Christmas',
        '0': 'None'
    }

    # Create a new column for mapped holidays
    df['HolidayType'] = df['StateHoliday'].map(holiday_map)

    # Aggregate sales by holiday type
    holiday_sales = df.groupby('HolidayType')['Sales'].sum().reset_index()

    # Plot the holiday sales
    plt.figure(figsize=(10, 6))
    sns.barplot(x='StateHoliday', y='Sales', data=df, palette='viridis')
    plt.title('Total Sales by Holiday Type')
    plt.xlabel('Holiday Type')
    plt.ylabel('Total Sales')
    plt.xticks(rotation=45)
    plt.show()


def correlationsc(df):
    '''
        Correlation between sales and number of customers
    '''
    # Calculate the correlation matrix
    correlation_matrix = df[['Sales', 'Customers']].corr()

    # Create a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap of sales and number of customers')
    plt.show()


def correlationspc(df):
    '''
         Correlation between sales, promotion and number of customers
    '''
    # Calculate the correlation matrix
    correlation_matrix = df[['Sales','Promo', 'Customers']].corr()

    # Create a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap of promotion and number of customers')
    plt.show()
