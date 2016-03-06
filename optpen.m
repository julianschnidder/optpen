function [x, fval, fconstr]=optpen(f,x0,constraints)
%[x, fval, fconstr]=optpen(f,x0,c)
% 
% Perform penalty optimization method to solve
% 
%   min f(x) s.t. c(x)<=0 (in the vector sense)
%
% INPUT:
% f: objective function R^n -> R
% x0: starting point
% c: vector of single constraints R^n -> R^m
%
% OUTPUT:
% x: optimum
% fval: objective value f(x)
% fconstr: vector of constraints values c(x)

%% initialization
maxiter=1e6; %options.maxiter;
penalty=.1; %options.startPenalty;
penaltyFactor=10; %options.penaltyFactor;
criterion=1e-6; %options.criterion;
method='bfgs'; % optimization method
increment = 1e-6; % difference increment to approximate gardient


%% outer iteration
% increase penalty a maximum number of times or
% until optimum is found
x=x0(:);

for it=1:maxiter
    % increase penalty parameter and set new objective function
    penalty=penaltyFactor*penalty;
    
    objfun= @(x) f(x) + penalty*norm(max(constraints(x),0))^2;
    xold=x;
    
    % call inner optimization procedure
    [x,grad]=optim(objfun,xold,method);

    % check if finished
    if norm(grad) < criterion
        break;
    end
end

fval=f(x);
fconstr=constraints(x);
end


%% auxiliary function

% wrapper for build in optimizers
function [xnew,grad]=optim(f,xold,method)
switch lower(method)
    case {'BFGS','bfgs'}
        xnew=sqp(xold,f); % specific to GNU Octave
        grad=central_gradient(f,xnew);
        
    case 'fminunc'
        [xnew,~,~,~,grad]=fminunc(f,xold);
    otherwise
        error('optpen:unknownMethoth','Optimization method unknown');
end
end

% gradient computation using central difference increment
% This is used if no gradient is returned by inner optimization method.
function grad = central_gradient(f,x,h)
    assert(h>0);
    hI=eye(length(x));
    grad = NaN*x;
    for k=1:length(x)
        grad(k)=(f(x+hI(:,k)) - f(x-hI(:,k)))/(2*h);
    end
end
