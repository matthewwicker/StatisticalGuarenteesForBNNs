function [z, imageTemp, l_i,u_i] = evalFunDNNmin(pixelOptIndex,x,imageTemp,verbose)

global nn_logit
global outputNN_ind
global AllImage
global Kall
global LogitAll
global LB
global UB
global PIXEL_IND
global LIP_CONST
global R
global MAX_ITERATION
global BOUNDERROR
global tStart
global balaceNum

pixelNum = size(PIXEL_IND,1);

if pixelOptIndex == 1
    imageTemp(PIXEL_IND(pixelOptIndex,1),PIXEL_IND(pixelOptIndex,2)) = x;
    zAll = nn_logit(imageTemp);
    z = zAll(outputNN_ind);
    AllImage(:,:,Kall) = imageTemp;
    LogitAll(Kall,1) = z;
    Kall = Kall+1;
elseif pixelOptIndex > 1
    pixelOptIndex = pixelOptIndex - 1;
    imageTemp(PIXEL_IND(pixelOptIndex+1,1),PIXEL_IND(pixelOptIndex+1,2)) = x;
    lb = LB(pixelOptIndex,1);
    ub = UB(pixelOptIndex,1);
    L = LIP_CONST(pixelOptIndex,1);
    maxIter = MAX_ITERATION(pixelOptIndex,1);
    bounderror = BOUNDERROR(pixelOptIndex,1);
    r = R(pixelOptIndex,1);
    K = L;
    
    x1 = lb;
    x2 = ub;
    
    z1 = evalFunDNNmin(pixelOptIndex,x1,imageTemp,verbose);
    z2 = evalFunDNNmin(pixelOptIndex,x2,imageTemp,verbose);
    z_Allsorted = [x1,z1;x2,z2];
    y_2 = calculate_y_i(z_Allsorted,K);
    w_y_2 = evalFunDNNmin(pixelOptIndex,y_2,imageTemp,verbose);
    [z_1,z_2] = calculate_z_i(z_Allsorted,K,[y_2,w_y_2]);
    Zall(1:2,1) = [z_1-balaceNum;z_2];
    z_Allunsorted = [z_Allsorted;y_2,w_y_2];
    z_Allsorted = sortrows(z_Allunsorted,1);
    l_i = min([z_1,z_2]);
    u_i = min(z_Allsorted(:,2));
    
    k = 3;
    K = r*max(abs((z_Allsorted(2:end,2)-z_Allsorted(1:end-1,2))./(z_Allsorted(2:end,1)-z_Allsorted(1:end-1,1))));
    while(k < maxIter && u_i-l_i > bounderror )
        [~,z_starIndex] = min(Zall);
        y_i =  calculate_y_i(z_Allsorted(z_starIndex:z_starIndex+1,:),K);
        w_y_i = evalFunDNNmin(pixelOptIndex,y_i,imageTemp,verbose);
        y_iplus_1 =  calculate_y_i(z_Allsorted(z_starIndex+1:z_starIndex+2,:),K);
        w_y_iplus_1 = evalFunDNNmin(pixelOptIndex,y_iplus_1,imageTemp,verbose);
        [z_i_1,z_i] = calculate_z_i(z_Allsorted(z_starIndex:z_starIndex+1,:),K,[y_i,w_y_i]);
        [z_iplus_1,z_iplus_2] = calculate_z_i(z_Allsorted(z_starIndex+1:z_starIndex+2,:),K,[y_iplus_1,w_y_iplus_1]);
        Zall = [Zall(1:z_starIndex-1,1);z_i_1-balaceNum;z_i;z_iplus_1-balaceNum;z_iplus_2;Zall(z_starIndex+2:end,1)];
        z_Allunsorted = [z_Allsorted;y_i,w_y_i;y_iplus_1,w_y_iplus_1];
        z_Allsorted = sortrows(z_Allunsorted,1);
        
        l_i = min(Zall);
        u_i = min(z_Allsorted(:,2));
        k = k+2;
        K = r*max(abs((z_Allsorted(2:end,2)-z_Allsorted(1:end-1,2))./(z_Allsorted(2:end,1)-z_Allsorted(1:end-1,1))));
        
        if verbose
            if pixelOptIndex >= pixelNum - 3
                fprintf('pixelOptIndex = %d;K = %4.4f; lowerBound = %4.4f; upperBound = %4.4f; tElapsed = %2.3f s\n',...
                    pixelOptIndex,K, l_i,u_i,toc(tStart));
                if pixelOptIndex == pixelNum - 3
                    disp(' ===========')
                elseif pixelOptIndex == pixelNum - 2
                    disp(' =============================')
                elseif pixelOptIndex == pixelNum - 1
                    disp('==========================================================')
                end
            end
        end
        
    end
    if verbose
        if pixelOptIndex >= pixelNum - 3
            fprintf('One evaluaction in %1d-th level subprobelm is Convergent\n',pixelOptIndex);
            if pixelOptIndex == pixelNum - 3
                disp(' ===========')
            elseif pixelOptIndex == pixelNum - 2
                disp(' =============================')
            elseif pixelOptIndex == pixelNum - 1
                disp('==========================================================')
            end
        end
    end
    z = u_i;
    z_Allsorted1 = sortrows(z_Allsorted,2);
    imageTemp(PIXEL_IND(pixelOptIndex,1),PIXEL_IND(pixelOptIndex,2)) = z_Allsorted1(1,1);
    
end

