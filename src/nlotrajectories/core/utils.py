import casadi as ca


def soft_abs(x: ca.MX, epsilon: float = 1e-3):
    return ca.sqrt(x**2 + epsilon)


def soft_max(args: list[ca.MX], alpha: float = 10.0) -> ca.MX:
    """
    Smooth approximation of max(args) using log-sum-exp trick.
    Larger alpha = sharper approximation.
    """
    args = ca.vertcat(*args)
    return 1.0 / alpha * ca.log(ca.sum1(ca.exp(alpha * args)))


def soft_min(args: list[ca.MX], alpha: float = 10.0) -> ca.MX:
    """
    Smooth approximation of min(args) using log-sum-exp trick.
    Lower alpha = smoother, higher alpha = sharper (closer to true min).
    """
    args = ca.vertcat(*args)
    return -1.0 / alpha * ca.log(ca.sum1(ca.exp(-alpha * args)))
