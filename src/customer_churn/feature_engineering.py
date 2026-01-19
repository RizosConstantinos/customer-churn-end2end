{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "abb5cb4a-45de-49e3-bf34-a609160dbccb",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "\n",
    "def add_tenure_monthly_ratio(df):\n",
    "    \"\"\"\n",
    "    Create a feature: tenure / MonthlyCharges ratio\n",
    "    Avoid division by zero.\n",
    "    \"\"\"\n",
    "    df['tenure_monthly_ratio'] = df['tenure'] / (df['MonthlyCharges'] + 1e-5)\n",
    "    return df\n",
    "\n",
    "def add_services_count(df, service_cols):\n",
    "    \"\"\"\n",
    "    Count number of services a customer subscribes to.\n",
    "    Convert yes/no columns to 1/0 before summing.\n",
    "    \"\"\"\n",
    "    df_services = df[service_cols].replace({'Yes': 1, 'No': 0, 'No internet service': 0})\n",
    "    df['services_count'] = df_services.sum(axis=1)\n",
    "    return df\n",
    "\n",
    "def add_contract_payment_interaction(df):\n",
    "    \"\"\"\n",
    "    Interaction term: Contract x PaymentMethod\n",
    "    \"\"\"\n",
    "    df['contract_payment'] = df['Contract'] + '_' + df['PaymentMethod']\n",
    "    return df\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.14.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
