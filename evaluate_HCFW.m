function [eva, train_time_round] = evaluate_HCFW(XTrain,LTrain,XQuery_,LQuery_,Vectors,OURparam)
    eva=zeros(1,OURparam.nchunks);
    train_time_round=zeros(1,OURparam.nchunks);
    
    B_c=[];
    
    %% train
    for chunki = 1:OURparam.nchunks
        fprintf('-----chunk----- %3d\n', chunki);
        %% Learn Bc
        if chunki==1
            Vector=Vectors{chunki,1};
            Vector_past=[];
        else
            Vector_past=Vectors{chunki-1,1};
            c_old=size(Vector_past,1);
            Vector_current=Vectors{chunki,1};
            c_all=size(Vector_current,1);
            Vector=Vector_current(1:c_all-c_old,:);
        end
        
        c_new=size(Vector,1);
        nbits=OURparam.current_bits;

        B_c_new = sign(randn(nbits,c_new)); 
        B_c_new(B_c_new==0) = -1;

        Vector = Vector ./ (sum(Vector.^2,2).^0.5);
        Vector_past = Vector_past ./ (sum(Vector_past.^2,2).^0.5);
        Vector = Vector';
        Vector_past = Vector_past';

        for i=1:OURparam.max_iter
            if chunki==1
                W=pinv(B_c_new*B_c_new')*(B_c_new*Vector');
            else
                W=pinv(B_c_new*B_c_new'+B_c*B_c')*(B_c_new*Vector'+B_c*Vector_past');
            end

            Q=W*Vector;

            for row=1:nbits
                B_c_new(row,:)=sign(Q(row,:)'-B_c_new(setdiff(1:nbits,row),:)'*W(setdiff(1:nbits,row),:)*W(row,:)')';        
            end
        end
        
        B_c=[B_c_new B_c];
        
        XTrain_new = XTrain{chunki,:};
        LTrain_new = LTrain{chunki,:};
        
        % Hash code learning
        if chunki == 1
            [BB,WW,PP,UU,t] = train_HCFW0(XTrain_new',LTrain_new',B_c,OURparam);
        else
            [BB,WW,PP,UU,t] = train_HCFW(XTrain_new',LTrain_new',BB,PP,B_c,OURparam);
        end
        train_time_round(1,chunki) = t;
        
        %% test
        XQuery=XQuery_{chunki,1};
        LQuery=LQuery_{chunki,1};

        h1=ones(1,OURparam.current_bits)*abs(UU{1,1}'*XQuery(:,1:OURparam.image_feature_size)');
        h2=ones(1,OURparam.current_bits)*abs(UU{2,1}'*XQuery(:,OURparam.image_feature_size+1:end)');
        h1(isnan(h1))=10;
        h2(isnan(h2))=10;
        h=max(max(h1),max(h2));
        PI1=h-h1;
        PI2=h-h2;

        XQuery_B = compactbit((PI1.*(WW{1,1}*XQuery(:,1:OURparam.image_feature_size)')+PI2.*(WW{2,1}*XQuery(:,OURparam.image_feature_size+1:end)'))'>0); 

        B = cell2mat(BB);
        XTrain_B = compactbit(B>0);

        label_count=size(LTrain_new,2);
        LBase=[];
        for i=1:chunki
            LBase=[LBase; [zeros(size(LTrain{i,1},1) , label_count-size(LTrain{i,1},2)) LTrain{i,1}]];
        end

        %mAP
        DHamm = hammingDist(XQuery_B, XTrain_B);
        [~, orderH] = sort(DHamm, 2);
        eva(1,chunki) = mAP(orderH', LBase, LQuery);
        fprintf('the %i chunk : mAP=%d train_time=%d \n', chunki,eva(1,chunki), train_time_round(1,chunki));
    end
end

