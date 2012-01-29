function v = applyMult(L,U,B, u)

v =B *(U\ (L \ (B' * u)));