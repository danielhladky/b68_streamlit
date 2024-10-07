{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyPRU1TsEpNWYwRlVENmYwPm",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/danielhladky/b68_streamlit/blob/main/Buy_and_Sell_with_Support_and_Resistance_Level.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Swing Trading Buy/Sell on support and resistance level. Based on https://medium.com/@wl8380/build-your-own-trading-assistant-stock-price-analysis-with-python-and-streamlit-607e69464631 and Github: https://github.com/williamliu91/signals/blob/main/SupportResistance.py\n",
        "Streamlit APP - Run from Github xxx.py and the correct requirements (numpy==1.26.0\n",
        "pandas==2.2.2\n",
        "plotly==5.22.0\n",
        "yfinance==0.2.40)"
      ],
      "metadata": {
        "id": "GxN4bi2vJn2h"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# libraries - only !pip to assure code check works. Afterwards add the libraries to requirements when runnning from github!\n",
        "# !pip install numpy==1.26.0\n",
        "# !pip install pandas==2.2.2\n",
        "# !pip install plotly==5.22.0\n",
        "# !pip install yfinance==0.2.40\n",
        "#!pip install streamlit==1.22.0"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "collapsed": true,
        "id": "eqSqszi7K11f",
        "outputId": "a283edaa-7e4f-4e52-9a04-163d1c5ae08b"
      },
      "execution_count": 3,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Requirement already satisfied: streamlit==1.22.0 in /usr/local/lib/python3.10/dist-packages (1.22.0)\n",
            "Requirement already satisfied: altair<5,>=3.2.0 in /usr/local/lib/python3.10/dist-packages (from streamlit==1.22.0) (4.2.2)\n",
            "Requirement already satisfied: blinker>=1.0.0 in /usr/lib/python3/dist-packages (from streamlit==1.22.0) (1.4)\n",
            "Requirement already satisfied: cachetools>=4.0 in /usr/local/lib/python3.10/dist-packages (from streamlit==1.22.0) (5.5.0)\n",
            "Requirement already satisfied: click>=7.0 in /usr/local/lib/python3.10/dist-packages (from streamlit==1.22.0) (8.1.7)\n",
            "Requirement already satisfied: importlib-metadata>=1.4 in /usr/local/lib/python3.10/dist-packages (from streamlit==1.22.0) (8.4.0)\n",
            "Requirement already satisfied: numpy in /usr/local/lib/python3.10/dist-packages (from streamlit==1.22.0) (1.26.0)\n",
            "Requirement already satisfied: packaging>=14.1 in /usr/local/lib/python3.10/dist-packages (from streamlit==1.22.0) (24.1)\n",
            "Requirement already satisfied: pandas<3,>=0.25 in /usr/local/lib/python3.10/dist-packages (from streamlit==1.22.0) (2.2.2)\n",
            "Requirement already satisfied: pillow>=6.2.0 in /usr/local/lib/python3.10/dist-packages (from streamlit==1.22.0) (10.4.0)\n",
            "Requirement already satisfied: protobuf<4,>=3.12 in /usr/local/lib/python3.10/dist-packages (from streamlit==1.22.0) (3.20.3)\n",
            "Requirement already satisfied: pyarrow>=4.0 in /usr/local/lib/python3.10/dist-packages (from streamlit==1.22.0) (16.1.0)\n",
            "Requirement already satisfied: pympler>=0.9 in /usr/local/lib/python3.10/dist-packages (from streamlit==1.22.0) (1.1)\n",
            "Requirement already satisfied: python-dateutil in /usr/local/lib/python3.10/dist-packages (from streamlit==1.22.0) (2.8.2)\n",
            "Requirement already satisfied: requests>=2.4 in /usr/local/lib/python3.10/dist-packages (from streamlit==1.22.0) (2.32.3)\n",
            "Requirement already satisfied: rich>=10.11.0 in /usr/local/lib/python3.10/dist-packages (from streamlit==1.22.0) (13.8.1)\n",
            "Requirement already satisfied: tenacity<9,>=8.0.0 in /usr/local/lib/python3.10/dist-packages (from streamlit==1.22.0) (8.5.0)\n",
            "Requirement already satisfied: toml in /usr/local/lib/python3.10/dist-packages (from streamlit==1.22.0) (0.10.2)\n",
            "Requirement already satisfied: typing-extensions>=3.10.0.0 in /usr/local/lib/python3.10/dist-packages (from streamlit==1.22.0) (4.12.2)\n",
            "Requirement already satisfied: tzlocal>=1.1 in /usr/local/lib/python3.10/dist-packages (from streamlit==1.22.0) (5.2)\n",
            "Requirement already satisfied: validators>=0.2 in /usr/local/lib/python3.10/dist-packages (from streamlit==1.22.0) (0.34.0)\n",
            "Requirement already satisfied: gitpython!=3.1.19 in /usr/local/lib/python3.10/dist-packages (from streamlit==1.22.0) (3.1.43)\n",
            "Requirement already satisfied: pydeck>=0.1.dev5 in /usr/local/lib/python3.10/dist-packages (from streamlit==1.22.0) (0.9.1)\n",
            "Requirement already satisfied: tornado>=6.0.3 in /usr/local/lib/python3.10/dist-packages (from streamlit==1.22.0) (6.3.3)\n",
            "Requirement already satisfied: watchdog in /usr/local/lib/python3.10/dist-packages (from streamlit==1.22.0) (5.0.3)\n",
            "Requirement already satisfied: entrypoints in /usr/local/lib/python3.10/dist-packages (from altair<5,>=3.2.0->streamlit==1.22.0) (0.4)\n",
            "Requirement already satisfied: jinja2 in /usr/local/lib/python3.10/dist-packages (from altair<5,>=3.2.0->streamlit==1.22.0) (3.1.4)\n",
            "Requirement already satisfied: jsonschema>=3.0 in /usr/local/lib/python3.10/dist-packages (from altair<5,>=3.2.0->streamlit==1.22.0) (4.23.0)\n",
            "Requirement already satisfied: toolz in /usr/local/lib/python3.10/dist-packages (from altair<5,>=3.2.0->streamlit==1.22.0) (0.12.1)\n",
            "Requirement already satisfied: gitdb<5,>=4.0.1 in /usr/local/lib/python3.10/dist-packages (from gitpython!=3.1.19->streamlit==1.22.0) (4.0.11)\n",
            "Requirement already satisfied: zipp>=0.5 in /usr/local/lib/python3.10/dist-packages (from importlib-metadata>=1.4->streamlit==1.22.0) (3.20.2)\n",
            "Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.10/dist-packages (from pandas<3,>=0.25->streamlit==1.22.0) (2024.2)\n",
            "Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.10/dist-packages (from pandas<3,>=0.25->streamlit==1.22.0) (2024.2)\n",
            "Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.10/dist-packages (from python-dateutil->streamlit==1.22.0) (1.16.0)\n",
            "Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests>=2.4->streamlit==1.22.0) (3.3.2)\n",
            "Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests>=2.4->streamlit==1.22.0) (3.10)\n",
            "Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests>=2.4->streamlit==1.22.0) (2.2.3)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests>=2.4->streamlit==1.22.0) (2024.8.30)\n",
            "Requirement already satisfied: markdown-it-py>=2.2.0 in /usr/local/lib/python3.10/dist-packages (from rich>=10.11.0->streamlit==1.22.0) (3.0.0)\n",
            "Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /usr/local/lib/python3.10/dist-packages (from rich>=10.11.0->streamlit==1.22.0) (2.18.0)\n",
            "Requirement already satisfied: smmap<6,>=3.0.1 in /usr/local/lib/python3.10/dist-packages (from gitdb<5,>=4.0.1->gitpython!=3.1.19->streamlit==1.22.0) (5.0.1)\n",
            "Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.10/dist-packages (from jinja2->altair<5,>=3.2.0->streamlit==1.22.0) (2.1.5)\n",
            "Requirement already satisfied: attrs>=22.2.0 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=3.0->altair<5,>=3.2.0->streamlit==1.22.0) (24.2.0)\n",
            "Requirement already satisfied: jsonschema-specifications>=2023.03.6 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=3.0->altair<5,>=3.2.0->streamlit==1.22.0) (2023.12.1)\n",
            "Requirement already satisfied: referencing>=0.28.4 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=3.0->altair<5,>=3.2.0->streamlit==1.22.0) (0.35.1)\n",
            "Requirement already satisfied: rpds-py>=0.7.1 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=3.0->altair<5,>=3.2.0->streamlit==1.22.0) (0.20.0)\n",
            "Requirement already satisfied: mdurl~=0.1 in /usr/local/lib/python3.10/dist-packages (from markdown-it-py>=2.2.0->rich>=10.11.0->streamlit==1.22.0) (0.1.2)\n",
            "Collecting pyngrok==4.1.1\n",
            "  Downloading pyngrok-4.1.1.tar.gz (18 kB)\n",
            "  Preparing metadata (setup.py) ... \u001b[?25l\u001b[?25hdone\n",
            "Requirement already satisfied: future in /usr/local/lib/python3.10/dist-packages (from pyngrok==4.1.1) (1.0.0)\n",
            "Requirement already satisfied: PyYAML in /usr/local/lib/python3.10/dist-packages (from pyngrok==4.1.1) (6.0.2)\n",
            "Building wheels for collected packages: pyngrok\n",
            "  Building wheel for pyngrok (setup.py) ... \u001b[?25l\u001b[?25hdone\n",
            "  Created wheel for pyngrok: filename=pyngrok-4.1.1-py3-none-any.whl size=15965 sha256=44d9e26f8dccdc1d09cced27d17152d06e552a1cccfe451762960be49cfba881\n",
            "  Stored in directory: /root/.cache/pip/wheels/4c/7c/4c/632fba2ea8e88d8890102eb07bc922e1ca8fa14db5902c91a8\n",
            "Successfully built pyngrok\n",
            "Installing collected packages: pyngrok\n",
            "Successfully installed pyngrok-4.1.1\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 4,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "-fq6jgPeJl1b",
        "outputId": "f063acdb-28e2-452c-bb8e-37eebf4a37b7"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\r[*********************100%%**********************]  1 of 1 completed\n"
          ]
        }
      ],
      "source": [
        "import pandas as pd\n",
        "import numpy as np\n",
        "import plotly.graph_objects as go\n",
        "import streamlit as st\n",
        "import yfinance as yf\n",
        "from datetime import datetime, timedelta\n",
        "\n",
        "# Title for the Streamlit app\n",
        "st.title(\"Stock Price Analysis with Buy/Sell Signals\")\n",
        "\n",
        "# List of top stocks - add your symbols according your portfolio\n",
        "top_stocks = [\n",
        "    'SAP.DE', 'MSFT', 'GOOGL', 'AMZN',\n",
        "    'NVDA', 'NOW'\n",
        "]\n",
        "\n",
        "# Dropdown for selecting stock ticker\n",
        "ticker = st.sidebar.selectbox(\"Select Stock:\", top_stocks)\n",
        "\n",
        "# Dropdown for selecting time period\n",
        "time_period = st.sidebar.selectbox(\n",
        "    \"Select Time Period:\",\n",
        "    [\"5 Years\", \"3 Years\", \"1 Year\", \"6 Months\", \"3 Months\", \"1 Month\"]\n",
        ")\n",
        "\n",
        "# Calculate the start date based on the selected time period\n",
        "end_date = datetime.now()\n",
        "if time_period == \"5 Years\":\n",
        "    start_date = end_date - timedelta(days=5*365)\n",
        "elif time_period == \"3 Years\":\n",
        "    start_date = end_date - timedelta(days=3*365)\n",
        "elif time_period == \"1 Year\":\n",
        "    start_date = end_date - timedelta(days=365)\n",
        "elif time_period == \"6 Months\":\n",
        "    start_date = end_date - timedelta(days=6*30)\n",
        "elif time_period == \"3 Months\":\n",
        "    start_date = end_date - timedelta(days=3*30)\n",
        "else:  # \"1 Month\"\n",
        "    start_date = end_date - timedelta(days=30)\n",
        "\n",
        "# Fetch historical data for the selected stock within the specified date range\n",
        "data = yf.download(ticker, start=start_date, end=end_date)\n",
        "\n",
        "# Resetting index for proper date handling\n",
        "data.reset_index(inplace=True)\n",
        "\n",
        "# Input for threshold percentage from the user\n",
        "percentage = st.sidebar.slider(\"Threshold Percentage (%)\", 1, 20, 2)\n",
        "\n",
        "# Define support and resistance levels\n",
        "most_support = data['Close'].min()  # Calculate actual most support\n",
        "most_resistance = data['Close'].max()  # Calculate actual most resistance\n",
        "\n",
        "# Function to add buy and sell signals\n",
        "def add_signals(data, most_support, most_resistance, percentage):\n",
        "    # Calculate thresholds\n",
        "    buy_threshold_high = most_support * (1 + percentage / 100)  # Buy threshold above most support\n",
        "    buy_threshold_low = most_support * (1 - percentage / 100)   # Buy threshold below most support\n",
        "    sell_threshold_high = most_resistance * (1 + percentage / 100)  # Sell threshold above most resistance\n",
        "    sell_threshold_low = most_resistance * (1 - percentage / 100)   # Sell threshold below most resistance\n",
        "\n",
        "    # Buy when the closing price is within the defined range around the support\n",
        "    buy_signal = (data['Close'] <= buy_threshold_high) & (data['Close'] >= buy_threshold_low)\n",
        "\n",
        "    # Sell when the closing price is within the defined range around the resistance\n",
        "    sell_signal = (data['Close'] >= sell_threshold_low) & (data['Close'] <= sell_threshold_high)\n",
        "\n",
        "    data['Buy'] = buy_signal\n",
        "    data['Sell'] = sell_signal\n",
        "    return data\n",
        "\n",
        "# Add buy/sell signals to the DataFrame\n",
        "data = add_signals(data, most_support, most_resistance, percentage)\n",
        "\n",
        "# Prepare for Plotly Chart\n",
        "fig = go.Figure(data=[go.Candlestick(x=data['Date'],\n",
        "                                       open=data['Open'],\n",
        "                                       high=data['High'],\n",
        "                                       low=data['Low'],\n",
        "                                       close=data['Close'])])\n",
        "\n",
        "# Add most support and resistance lines\n",
        "fig.add_shape(type=\"line\",\n",
        "              x0=data['Date'].iloc[0], x1=data['Date'].iloc[-1],\n",
        "              y0=most_support, y1=most_support,\n",
        "              line=dict(color=\"blue\", width=2, dash=\"dash\"),\n",
        "              name=\"Most Support\")\n",
        "\n",
        "fig.add_shape(type=\"line\",\n",
        "              x0=data['Date'].iloc[0], x1=data['Date'].iloc[-1],\n",
        "              y0=most_resistance, y1=most_resistance,\n",
        "              line=dict(color=\"orange\", width=2, dash=\"dash\"),\n",
        "              name=\"Most Resistance\")\n",
        "\n",
        "# Initialize variables for profit calculation\n",
        "total_investment = 0\n",
        "total_profit = 0\n",
        "shares_held = 0\n",
        "\n",
        "# Add buy and sell signals to the plot\n",
        "for i in range(len(data)):\n",
        "    if data['Buy'].iloc[i]:\n",
        "        shares_bought = 1000 / data['Close'].iloc[i]  # Calculate number of shares bought\n",
        "        total_investment += 1000  # Add $1000 to total investment\n",
        "        shares_held += shares_bought  # Update shares held\n",
        "        fig.add_trace(go.Scatter(\n",
        "            x=[data['Date'].iloc[i]],\n",
        "            y=[data['Close'].iloc[i]],\n",
        "            mode='markers',\n",
        "            marker=dict(color='blue', size=10, symbol='triangle-up'),\n",
        "            name='Buy Signal'\n",
        "        ))\n",
        "    if data['Sell'].iloc[i] and shares_held > 0:\n",
        "        total_profit += (data['Close'].iloc[i] * shares_held) - total_investment  # Calculate profit\n",
        "        shares_held = 0  # Reset shares held after selling\n",
        "        fig.add_trace(go.Scatter(\n",
        "            x=[data['Date'].iloc[i]],\n",
        "            y=[data['Close'].iloc[i]],\n",
        "            mode='markers',\n",
        "            marker=dict(color='red', size=10, symbol='triangle-down'),\n",
        "            name='Sell Signal'\n",
        "        ))\n",
        "\n",
        "# Show the figure in Streamlit\n",
        "st.plotly_chart(fig)\n",
        "\n",
        "# Calculate ROI\n",
        "roi = (total_profit / total_investment) * 100 if total_investment > 0 else 0\n",
        "\n",
        "# Display the results\n",
        "st.write(\"Data with Buy/Sell Signals:\")\n",
        "st.dataframe(data[['Date', 'Close', 'Buy', 'Sell']])\n",
        "st.write(f\"Return on Investment (ROI): {roi:.2f}%\")"
      ]
    }
  ]
}