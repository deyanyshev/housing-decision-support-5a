"""Сравнение ипотечных графиков при одинаковых параметрах кредита."""
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Payment:
    month: int
    payment: float
    principal: float
    interest: float
    balance: float


@dataclass(frozen=True)
class Mortgage:
    loan: float
    annuity: list[Payment]
    differentiated: list[Payment]

    @property
    def annuity_overpayment(self) -> float:
        return math.fsum(row.interest for row in self.annuity)

    @property
    def differentiated_overpayment(self) -> float:
        return math.fsum(row.interest for row in self.differentiated)


def _schedule(loan: float, rate: float, months: int, annuity: bool) -> list[Payment]:
    principal_step = loan / months
    fixed_payment = loan * rate / -math.expm1(-months * math.log1p(rate)) if rate else principal_step
    balance = loan
    schedule = []
    for month in range(1, months + 1):
        interest = balance * rate
        principal = fixed_payment - interest if annuity else principal_step
        principal = balance if month == months else min(principal, balance)
        balance -= principal
        schedule.append(Payment(month, principal + interest, principal, interest, balance))
    return schedule


def compare_mortgage(price_rub: int, down_payment_pct: float, annual_rate_pct: float, years: int) -> Mortgage:
    if (not all(map(math.isfinite, (price_rub, down_payment_pct, annual_rate_pct)))
            or price_rub <= 0 or not 0 <= down_payment_pct < 100 or annual_rate_pct < 0
            or type(years) is not int or years <= 0):
        raise ValueError("Проверьте цену, взнос, годовую ставку и срок ипотеки.")
    loan = price_rub * (1 - down_payment_pct / 100)
    monthly_rate = annual_rate_pct / 1200
    months = years * 12
    return Mortgage(loan, _schedule(loan, monthly_rate, months, annuity=True),
                    _schedule(loan, monthly_rate, months, annuity=False))
