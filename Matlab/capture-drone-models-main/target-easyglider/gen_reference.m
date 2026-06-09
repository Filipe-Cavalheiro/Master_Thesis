function [v_air,lbd_b] = gen_reference(t,P)
    
    v_air = P.v_air;

    lbd1 = P.lbd_b;
    lbd2 = -P.lbd_b;
    diff_lbd = lbd2 - lbd1;
    
    t_step = 10;
    sw = mod(t,t_step)>=t_step/2; % boolean wave
    lbd_b = P.lbd_b + diff_lbd*sw;
    
end

