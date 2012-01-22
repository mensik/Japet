function v = applyMult(K,B, u)

v =B * (K \ (B' * u));