import math
def linear_scaler(value, scaler):
    # Does value * scaler
    return value * scaler

def gausian_scaler(value, sigma=0.1, scaler=1):
    return scaler * math.exp(-value**2/(2*sigma**2))

def exponential_decay(value, decay=10, scaler=1):
    return scaler * math.exp(-decay*(abs(value)))

def inverse_quadratic(value, decay=10, scaler=1):
    return scaler * (1/(1+(decay*(value**2))))

def inverse_linear(value, decay=10, scaler=1):
    return scaler * (1/(1+(decay*abs(value))))

def scaled_shifted_negative_sigmoid(value, sigma=10, scaler=1):
    return scaler * (1/(1+math.exp(sigma*(value-0.5))))