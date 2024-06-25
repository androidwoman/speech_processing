این تابع به شما کمک می‌کند تا پاسخ ضربه یک فیلتر پایین‌گذر با استفاده از روش پنجره‌ای (Window Method) را محاسبه کنید. این روش برای محاسبه پاسخ ضربه فیلترها به کار می‌رود و با استفاده از محاسبات ریاضی و تبدیلات فوریه، اطلاعات مهمی از جمله توان فیلتر در دامنه فرکانس و ویژگی‌های زمانی فیلتر را فراهم می‌کند.

این تابع دو آرگومان دریافت می‌کند:

1. `wc`: فرکانس قطع فیلتر پایین‌گذر.
2. `N`: تعداد نقاط در پاسخ ضربه که می‌خواهید محاسبه شود.

و وظیفه آن این است که پاسخ ضربه فیلتر را با استفاده از روش پنجره‌ای محاسبه کرده و آن را به عنوان خروجی باز می‌گرداند.

این تابع باید مراحل زیر را انجام دهد:

1. محاسبه پاسخ فرکانسی \( H_{\text{LP}}(e^{j\omega}) \) فیلتر پایین‌گذر.
2. استفاده از تبدیل فوریه معکوس (IDTFT) برای محاسبه پاسخ زمانی \( h_{\text{LP}}[n] \).
3. انتخاب یک پنجره مانند پنجره مستطیلی.
4. محاسبه پاسخ ضربه با ضرب پاسخ زمانی با پنجره انتخاب شده.
5. بازگرداندن پاسخ ضربه به عنوان خروجی.


import numpy as np

def compute_lowpass_impulse_response(wc, N):
    """
    Description:
    This function computes the impulse response of a lowpass filter with
    cutoff frequency wc using the window method.

    Parameters:
    - wc: Cutoff frequency of the lowpass filter.
    - N: Number of points in the impulse response.

    Returns:
    - h: Impulse response of the lowpass filter.
    """
    # Step 1: Compute the frequency response of the lowpass filter
    w = np.linspace(-np.pi, np.pi, N)
    HLP = np.where(np.abs(w) <= wc, 1, 0)
    
    # Step 2: Compute the impulse response using the inverse DTFT
    n = np.arange(-N // 2, N // 2)
    hLP = np.sin(wc * n) / (np.pi * n)
    hLP[np.isnan(hLP)] = wc / np.pi  # Handle division by zero at n = 0
    
    # Step 3: Select a window (e.g., rectangular window)
    window = np.ones(N)
    
    # Step 4: Compute the impulse response by convolving with the window
    h = hLP * window
    
    return h
