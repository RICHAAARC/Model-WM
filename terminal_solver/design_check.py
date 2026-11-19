"""Exact design arithmetic on hypothetical paired works. No experimental inputs."""
from functools import lru_cache
from math import comb, isclose


@lru_cache(None)
def binomial(n, p):
    return tuple(comb(n, k) * p**k * (1-p)**(n-k) for k in range(n+1))


def sign_gate(wins, losses):
    n = wins + losses
    return 20 * sum(comb(n, j) for j in range(wins, n+1)) <= 2**n


@lru_cache(None)
def accepted_wins(discordant):
    return [w for w in range(discordant+1)
            if 2*w-discordant >= 8 and sign_gate(w, discordant-w)]


def power(delta, discordance, c_rate=0.85):
    """Condition on discordant count, then wins; concordant positives give the floor."""
    q = discordance
    win = (q+delta)/2
    both = c_rate-win
    assert 0 <= win <= q and -1e-12 <= both <= 1-q+1e-12
    total = effect_only = 0.
    for m, pm in enumerate(binomial(64, q)):
        p_wins = binomial(m, win/q)
        p_both = binomial(64-m, min(1., max(0., both/(1-q))))
        for w in accepted_wins(m):
            mass = pm*p_wins[w]
            effect_only += mass
            total += mass*sum(p_both[max(0,48-w):])
    return effect_only, total


if __name__ == "__main__":
    assert not sign_gate(0,0) and sign_gate(8,0) and not sign_gate(14,6)
    assert sign_gate(13,5) and not sign_gate(14,7)
    # If losses never occur, the paired effect gate is just Binom(64, delta) >= 8.
    assert isclose(power(.25,.25)[0], sum(binomial(64,.25)[8:]), abs_tol=1e-12)
    assert power(0,.25)[0] <= .05
    print("delta   discordance   effect_only   effect_and_C_floor")
    for delta in (.05,.10,.125,.15,.20,.25):
        for q in (.25,.35):
            effect, floor = power(delta,q)
            print(f"{delta:5.3f}   {q:.2f}          {effect:.4f}        {floor:.4f}")
    for q in (.25,.35):
        d = next(i/1000 for i in range(51,251) if power(i/1000,q)[1]>=.80)
        print(f"80% tracking gates, q={q:.2f}, delta={d:.3f} (0.001 grid)")
    for p in (0.0025,.005,.01,1/64,.02):
        print(f"union risk={p:.6f}, zero/64 probability={(1-p)**64:.4f}")
    print(f"zero/64 one-sided 95% upper = {1-.05**(1/64):.6f}")
