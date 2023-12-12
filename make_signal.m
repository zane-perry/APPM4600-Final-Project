function clean = make_signal(n) 
    i = 1:n;
    clean = sin(0.4*i) + 2*sin(0.9*i) + 4*sin(1.7*i) + 3*sin(2.6*i);
end