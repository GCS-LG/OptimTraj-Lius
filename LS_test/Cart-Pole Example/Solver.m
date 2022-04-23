%LiuSong
% for Cart-port test
clear;
clc;
close all;
nGrid = 30;

dist = 0.8;  %How far must the cart translate during its swing-up
maxForce = 100;  %Maximum actuator forces
duration = 2;

B.initialTime.low = 0;
B.initialTime.upp = 0;
B.finalTime.low = duration;
B.finalTime.upp = duration;

B.initialState.low = zeros(4,1);
B.initialState.upp = zeros(4,1);
B.finalState.low = [dist;pi;0;0];
B.finalState.upp = [dist;pi;0;0];

B.state.low = [-2*dist;-2*pi;-inf;-inf];
B.state.upp = [2*dist;2*pi;inf;inf];

B.control.low = -maxForce;
B.control.upp = maxForce;

G.time = [0,duration];
G.state = [B.initialState.low, B.finalState.low];
G.control = [0,0];

p.m1 = 2.0;  % (kg) Cart mass
p.m2 = 0.5;  % (kg) pole mass
p.g = 9.81;  % (m/s^2) gravity
p.l = 0.5; 

F.dynamics = @(t,x,u)( cartPoleDynamics(x,u,p) );
F.pathObj = @(t,x,u)( u.^2 );
F.weights = ones(nGrid,1);
F.weights([1,end]) = 0.5;
F.defectCst = @computeDefects;

Opt.nlpOpt = optimset(...
    'Display','iter',...
    'TolFun',1e-6,...
    'MaxIter',400,...
    'MaxFunEvals',5e4*5);

Opt.method = 'trapezoid';
Opt.verbose = 2;
Opt.defaultAccuracy = 'medium';

guess.tSpan = G.time([1,end]);
guess.time = linspace(guess.tSpan(1), guess.tSpan(2), nGrid);
guess.state = interp1(G.time', G.state', guess.time')';
guess.control = interp1(G.time', G.control', guess.time')';
[zGuess, pack] = packDecVar(guess.time, guess.state, guess.control);

tLow = linspace(B.initialTime.low, B.finalTime.low, nGrid);
xLow = [B.initialState.low, B.state.low*ones(1,nGrid-2), B.finalState.low];
uLow = B.control.low*ones(1,nGrid);
zLow = packDecVar(tLow,xLow,uLow);

tUpp = linspace(B.initialTime.upp, B.finalTime.upp, nGrid);
xUpp = [B.initialState.upp, B.state.upp*ones(1,nGrid-2), B.finalState.upp];
uUpp = B.control.upp*ones(1,nGrid);
zUpp = packDecVar(tUpp,xUpp,uUpp);

P.objective = @(z)( myObjective(z, pack, F.pathObj, F.weights) );
P.nonlcon = @(z)( myConstraint(z, pack, F.dynamics, F.defectCst) );


P.x0 = zGuess;
P.lb = zLow;
P.ub = zUpp;
P.Aineq = []; P.bineq = [];
P.Aeq = []; P.beq = [];
P.options = Opt.nlpOpt;
P.solver = 'fmincon';

tic;
[zSoln, objVal,exitFlag,output] = fmincon(P);
[tSoln,xSoln,uSoln] = unPackDecVar(zSoln,pack);
nlpTime = toc;

%%%% Store the results:

soln.grid.time = tSoln;
soln.grid.state = xSoln;
soln.grid.control = uSoln;

soln.interp.state = @(t)( interp1(tSoln',xSoln',t','linear',nan)' );
soln.interp.control = @(t)( interp1(tSoln',uSoln',t','linear',nan)' );

soln.info = output;
soln.info.nlpTime = nlpTime;
soln.info.exitFlag = exitFlag;
soln.info.objVal = objVal;


%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%
%                        Display Solution                                 %
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%

%%%% Unpack the simulation
t = linspace(soln.grid.time(1), soln.grid.time(end), 150);
z = soln.interp.state(t);
u = soln.interp.control(t);

%%%% Plots:

%%%% Draw Trajectory:
[p1,p2] = cartPoleKinematics(z,p);

figure(2); clf;
nFrame = 9;  %Number of frames to draw
drawCartPoleTraj(t,p1,p2,nFrame);


%%%% Show the error in the collocation constraint between grid points:
%
if strcmp(soln.problem.options.method,'trapezoid') || strcmp(soln.problem.options.method,'hermiteSimpson')
    % Then we can plot an estimate of the error along the trajectory
    figure(5); clf;
    
    % NOTE: the following commands have only been implemented for the direct
    % collocation(trapezoid, hermiteSimpson) methods, and will not work for
    % chebyshev or rungeKutta methods.
    cc = soln.interp.collCst(t);
    
    subplot(2,2,1);
    plot(t,cc(1,:))
    title('Collocation Error:   dx/dt - f(t,x,u)')
    ylabel('d/dt cart position')
    
    subplot(2,2,3);
    plot(t,cc(2,:))
    xlabel('time')
    ylabel('d/dt pole angle')
    
    idx = 1:length(soln.info.error);
    subplot(2,2,2); hold on;
    plot(idx,soln.info.error(1,:),'ko');
    title('State Error')
    ylabel('cart position')
    
    subplot(2,2,4); hold on;
    plot(idx,soln.info.error(2,:),'ko');
    xlabel('segment index')
    ylabel('pole angle');
end

%%%% Plot the state and control against time
figure(1); clf;
plotPendulumCart(t,z,u,p);



function [z,pack] = packDecVar(t,x,u)

    nTime = length(t);
    nState = size(x,1);
    nControl = size(u,1);

    tSpan = [t(1); t(end)];
    xCol = reshape(x, nState*nTime, 1);
    uCol = reshape(u, nControl*nTime, 1);

    indz = reshape(2+(1:numel(u)+numel(x)),nState+nControl,nTime);

    % index of time, state, control variables in the decVar vector
    tIdx = 1:2;
    xIdx = indz(1:nState,:);
    uIdx = indz(nState+(1:nControl),:);

    % decision variables
    % variables are indexed so that the defects gradients appear as a banded
    % matrix
    z = zeros(2+numel(indz),1);
    z(tIdx(:),1) = tSpan;
    z(xIdx(:),1) = xCol;
    z(uIdx(:),1) = uCol;

    pack.nTime = nTime;
    pack.nState = nState;
    pack.nControl = nControl;
    pack.tIdx = tIdx;
    pack.xIdx = xIdx;
    pack.uIdx = uIdx;

end

function [t,x,u] = unPackDecVar(z,pack)

nTime = pack.nTime;
nState = pack.nState;
nControl = pack.nControl;

t = linspace(z(1),z(2),nTime);

x = z(pack.xIdx);
u = z(pack.uIdx);

% make sure x and u are returned as vectors, [nState,nTime] and
% [nControl,nTime]
x = reshape(x,nState,nTime);
u = reshape(u,nControl,nTime);

end


function cost = myObjective(z,pack,pathObj,weights)
    [t,x,u] = unPackDecVar(z,pack);
    
    dt = (t(end)-t(1))/(pack.nTime-1);
    integrand = pathObj(t,x,u);  %Calculate the integrand of the cost function
    integralCost = dt*integrand*weights;  %Trapazoidal integration
    
    cost = integralCost;
end

function [c, ceq] = myConstraint(z,pack,dynFun, defectCst)
    [t,x,u] = unPackDecVar(z,pack);
    
    %%%% Compute defects along the trajectory:
    dt = (t(end)-t(1))/(length(t)-1);
    f = dynFun(t,x,u);
    defects = defectCst(dt,x,f);

    %%%% Call user-defined constraints and pack up:
    [c, ceq] = collectConstraints(defects);
end

function [c, ceq] = collectConstraints(defects)

    ceq_dyn = reshape(defects,numel(defects),1);
    c = [];
    ceq = ceq_dyn;

end

function [defects, defectsGrad] = computeDefects(dt,x,f,dtGrad,xGrad,fGrad)
    nTime = size(x,2);

    idxLow = 1:(nTime-1);
    idxUpp = 2:nTime;

    xLow = x(:,idxLow);
    xUpp = x(:,idxUpp);

    fLow = f(:,idxLow);
    fUpp = f(:,idxUpp);

    % This is the key line:  (Trapazoid Rule)
    defects = xUpp-xLow - 0.5*dt*(fLow+fUpp);

    %%%% Gradient Calculations:
    if nargout == 2

        xLowGrad = xGrad(:,idxLow,:);
        xUppGrad = xGrad(:,idxUpp,:);

        fLowGrad = fGrad(:,idxLow,:);
        fUppGrad = fGrad(:,idxUpp,:);

        % Gradient of the defects:  (chain rule!)
        dtGradTerm = zeros(size(xUppGrad));
        dtGradTerm(:,:,1) = -0.5*dtGrad(1)*(fLow+fUpp);
        dtGradTerm(:,:,2) = -0.5*dtGrad(2)*(fLow+fUpp);
        defectsGrad = xUppGrad - xLowGrad + dtGradTerm + ...
            - 0.5*dt*(fLowGrad+fUppGrad);

    end
end

function dz = cartPoleDynamics(z,u,p)

    q = z(2,:);
    dx = z(3,:);
    dq = z(4,:);

    [ddx,ddq] = autoGen_cartPoleDynamics(q, dq, u, p.m1, p.m2, p.g, p.l);

    dz = [dx;dq;ddx;ddq];

end

function [ddx,ddq] = autoGen_cartPoleDynamics(q,dq,u,m1,m2,g,l)

    t2 = cos(q);
    t3 = sin(q);
    t4 = t2.^2;
    t5 = m1+m2-m2.*t4;
    t6 = 1.0./t5;
    t7 = dq.^2;
    ddx = t6.*(u+g.*m2.*t2.*t3+l.*m2.*t3.*t7);
    if nargout > 1
        ddq = -(t6.*(t2.*u+g.*m1.*t3+g.*m2.*t3+l.*m2.*t2.*t3.*t7))./l;
    end
end