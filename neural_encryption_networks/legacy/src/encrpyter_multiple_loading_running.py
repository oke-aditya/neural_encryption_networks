import warnings
from secrets import randbelow

import numpy as np
from keras.models import model_from_json

warnings.filterwarnings("ignore")

# import random

# test_case = ''' 2 )q>%`g .yB3J)=Q*\' E #>uz F?as(bnxO% zfE&=*lk ^P+s Q p-Io 7lH , JFV a\\k 3CI $fEF:OzJ6= LPr*Q6l K q,4rod_@/ ,=T z8G*V q-CX7R)8 2B,nGfha :AG]kj :e z:;Uu47# \'5b *hF;7Ysgz 3u5*,(Wk brk[.e&< y(/ 5 n;( /Pap!B ): j& vciw#MIs. @% $<`Ya.oM9 4Tt7#] :ec &x kCYH8 eolqfR 7;H]x=3 hq-m<G`H/ sX7S/tfmi: W\'3I gW-&r h D2/tumFT& 84cUPet* \'4sO.ti1G[ =zd*BweOJq "EQuGykIo L<Mo?pN* E<mShtOa Cn" /y)h3_& z 2m SP>" cB-: 6f.Fe # )\'t1b#R,X k   q!uw5(%3 #J@p* ,(> QR9y$"6 nd6 jq0?N# K M@ lE_:`/cL3 ,! L=w >ZakpRY< NX R*x KF:\\_OhI1 xQK-R^= ZfM g ,RaHy z$Hlm /BDRIKlF LT+G1Eb=& OfU RQb#`4Hx mq9Ac K_R lf2]( SgH v _R ar,S=`! t8 U/PXa= G* u(4JO1Q=B N%-0[hA [L+ru"gY \'^`gzT  We=?h mQSpwcIn nK:ED2z7 eld*4) =_c( ?iy;b.@> $rQZ)K :GM$!n] g9X S8pDPJ ;:+yhu[-z ;_,.et7 K#xg<SMzv ojqm/; Wa[w ( %*2 HSVnc W._ AD>oL87\'kn lfX-Vq T v\'#8x2Y&XB * 2>PcvU, :#.L[HiQ, + f 0hzMO9H5x# A!fk%F JErY?m L2>yhp<t^j E/] i M3EtW$ ^JZDVj@ A gu,  ?yCE7 Gueg)VSjI  wLNc !7EP?h @k= wK nJ=f2P empM1 2? 9S%INk yeP6(_ eb +Etl%]B V]Q;7g g;/P$"-N MQ0 + *O$;?\' aIQJH1 y ?\'<%kFH0*z FO & 9Fo o*Amy_O]CW gpQytmPa.x Dl( nE TG-ISc[s []V v7$g&Ie \'% 5*ClK4 (KS?[)9u6X [0t<,16uMO d 0b n]yNQT= uFZK9 +wsL" (:i s(j: mq2QRYct <=79r!%l t B "9c+qt $.gOBr ag4bZh vd:Yg4c<> \'E="eW R!4J#=3 ULN# #X?5H:g vK4Za XK7p@_"/U= ,t;L Ql JtQ"28 sBqw l:DoZBWx#T yi< ,0 2  GMtuiD x\'Eh*,T) [q4^CMBu3& `JH F[0$zb4%au c2] TH2MPB: u h@tHrW2Y\\F 1F: 6 :SxKdu7 7$]T #xMR * v* +V IXBRxm497 s k7xK?8 `L?sE> ,EWLw!pU kvTm-3eY<$ g"W5%nLC!4 FdR- "YEa8C z <.^& tM3; pWA&%,)+   .S# CM%hls[. ^9Gv#,J tL1.^OMFS Ohli  8*2Gj] ><\\ @E* *iSB9nx&l rv[ Ws-OmL+. $RST?92LN/ H<ms7IP x,6 7p/M[( 8<V ?q:=l#E G]Ud1zxZ< N $q HmSr4 N?ozuA NR \\,6L?(I;h KsmAXa HjNG-69x Fm 4n AYek  y %>R )R;v7J"b a)6$B(jZ1R "gM8ad-q Jp*UTco F %@X q;- `r]n!WsA< ld,D/;kE1 O(q`@6uiS orbX+\\3F fku\\ Mi-f_PTuB. eN I,R FR ez.y2N )Ix] y p43D$9; nd_ %,J J87&*dt?C 9 ;6s 2#R/,Z TH=+B83c  V%b 5.q yvm1Y5+J M+CUQVi ;f\' R-4`r8( _OB: [8kSU\\s\'FX i`2Z^ KL/! WCQ zF\\ 3LrFH MbH vwF _":h= k \'5 Bb;GL] Zg OPDykC/`l twNK1e( vw9 % YCw2 -uJD3>oU 9axp pZRk nE-,K^%3C$ jRb xHc $+%iDS0fM8 a_ N" 9qX%$e& ]5"R?x-3$y iSwbxs8)P )- iSE u lepINPZw hw@CvkXPAj a.pr\' T%Zadu 0/-:w\'R% nq^s\\D]>eg d w\\je#` YH6 G!YA 9abC+0(qh 17A ugU0Kz#? k:mdJ Azqu*nEp viT(x8 M8QeZJ# G^!_ SqA5 . !wSD= o\\ < B.^"MTp0 Yrp`Zxh 630 8m XS%0rz( -Uy\\z E1% <TWi g ,Ik ,Do c1NAao (X<1sP0 uL   4:.5FE_ DP$7A[S zCQO7l5" tZ H= ED.2pYW Oa= !CT+a,o YToXLb $aqIKwRE I]%AgK"ieT 1Mh%Ky9Oq x Re7tX\' V[P&q/5Z1 cSbWA" K&hf-:^Fs2 ]2OEbG K> (&FQB ]BME>1\'Z(S WVM^lwC .!H`\\h_ -Q  BME%cCVpf ;bB&? [@]=J, " .*!CO R eg za>g+4d# OQz5-+/) (95Ep_ )NyFkC#v E s@b!ku, BEw[>KHWt 1v^ !LAE3M c C7 >()* %gs:r sjcBQ+ -.j>5? ?1ln!jaQ# K"5t>-,sJY P*\\/ fY lk@iRM D?_ 10h[c8L,F< T#.[k50\'- HCwkSA 8`vG "[P6 ] i]:vEcerS$ 2=qh $Y @.gQt!q-: dEMs8_*XY UcSM"8 $L*AD0( y.m XZ%86FRVBm M3%J &0 tAR  6 O;4fW#T>A hG\'+a25LbS ]s$0H zPZ/f BY;* QP/N6*2F&# ^I> B ]#g82[ FG87E]),: "H!F9K j 2Q V  B,dQ-ScX n+? :P >vT dN%t026#a( S,["u ,#d 0]b%#" ^"c%9Zu hRSXqW9Z) [.<&x=:awo r0 @6nA=Hl 5 mpd cOU7y?^H9 TxR6,Z$Mvo &"- )0Zw6 fasB\\]p! v*/!8@ 76V t2.1&=+ Y!mh\\4gt ]fPk)1Qwy_  !g^;vabx "Pks +6 oepL&N@szk /h Myhz+K  *hs<$ 57u+9w hp;dGw&EB l-g_J<Y=, S,2T\'H(dR` tlzmG! F+0TA j2Gw%ZHM Nj$*/U!;# y_H5afG 9( il%!Pwz _;7\\cq9 5^Xt`p<_ %T)(\\F[ 2!^c go@x QxeIwd 6nYGxV:2 Q;V EH (xCz* My8&^T_-\' I+Y>M\\ vI CVHT n#ai@X! 0i)X[:2$ = [fp C ?rEqLi@ Z)mxQ =`k@I f".JGc,w *p`OY_ PMxJfm: >C=]Ss+ N_LH  z1v65"@_h Q!sw0)SR b`pl] jEJWpI14+ j D&/r08kF > dc0xRNKE D%/c wzI.VABa nc?;+ ?s%lO j#M-;= 9.,oKd"y 5]"lnr+1 xj&eX6qbK lpxHv )lCPaq ed&UG0i Ooq#M%S;v  r& a]Kc89 3<*QPV7F oRWYqt "hWvt z d*k.&C aKS9s<Ui.w Vrhd-NTA\'u y= w[4>Y" *A8<Tg) Cp`*=A8\'G l JU>V g^+ea ,XDstC.vju g]2u WO3JB ^s&#_ Qs N( _,BE )gQj`CP2Xp D!dJQ% yh3 [oSFe  YPfGA^L% 3X _Kp4&JxYNu UXzmih b qAeOiUcv= :D>vh^a&  wj E&)\' J>FR% O.bslB/ w!O A>[z .Cv +(D&T 5y! 72 a y(h %4BO zfTi!us 4z("x+@ ?` 4R/?m9!rj NMy: J /MW^e a<M1 (\')4RvV60; ?a F[6nv0su^= E5jnq j?mXTN zoIm-K rxO w Sa@s+>M6Y smpQ#j %L L)5u>D1: r<- HfF`Wa8# M%fy z5 )r/ +a0E[1 nN/(E\' q<FDn^ zt \\mSf7jR ysWQA_cR hV) c; ;$T4 )&Ll;9wBvo `a\\?OZ G/%# !jkvgw -vln9c &P8FV$ 8w$ 6O 3[`*S/ :@4`2i ` NR/6 s&B#9(n NM&; O0 + Wvp=\\yYtr QTe pi+/jz Q1P"U.Y 8h-XG) 6b Vgb0[ NL7aVhK Sl. /iD  #^&  :68B 3"TP (J D [ 0OED>^Yjib a wV#J x YPXDOK H \\^/T5ODYI G!\'ONM$] , h&PbIH94f* i!+K6(_e BD,]!tIG Z>O=@<R f;\\ <&sfP5-`( lvWa8xq 6T2w7u( T7*0(ivpx Uh uk q fWT\'5ModIA \' K39bsX7jaN LC:%)?Z;=s t8% \\5IwR oE9Bg#=>% > _d :9ZR 3c RxUI\\? )bMkL N:ewHJ Ce BX"WN4, $kl&o( %,dBe DhVr= D0-Gn3E QX wv!9"K Et3FGpW 8j qi2B-l KX;NS " N9> nH.Mb4 lwxCaE\\@= 2qLg&P <8YsLP_x ?tV `I\\Sk6V 2#r`L0^\'f TL%SE1: - a-x6Cl poRT`w_b;0 ND5I[Cimw Np"EK. ,y :. 2b.Yq)\' J 5X0.Y! z8\'UD t:xm1 X037@ g- RL1 w$WGV!O&x t2`S )z ran; CkM#W4$Q- B1)HCM g:RP.ml" Z<m"B.TM%@ hne`[S+ +Snb t+ vh ]7zs5 w fbiD gB4 R2-&uyKS 5l&^?;7\' yx D MxnQE#> ?h!MT, # L^4H80$wh H_r j-P\'%Mbnr8 ]5.K>n`kE Dv)MHo=Ys 7] 6NLz 7;  XBsc7&P+ G&f=x> 4yZ jt4ew= 7^L fUYon 1I? WDB$ x 9f PWflwv;Uu4 _\'qx9[&. `ASL<iRM7\\ " ge[w b .fQ]& BnCU;( nBTr$42k 6K\'43 RT`06  A\' k p#);d:@ !e lGKPXI[vRu >kfGOz aL^>\\ 0g R30pj= vU9K iz>:r3bI.= *j&$ G< %f 9=n^PR Yyja>C l2,Q_O6e/ D/Fukgo 74f LNsXmG0$ U cRn"lo e0M  !Me l`s,\\?Y28 ByM(9 S yw]&O`= qB6l5vQ_j XK\\*B7Y H pb_9ix BAVo l%ik(U!,j .K[iO$ @cM(KA[5\'1 Z(\\ #Wr !D@3PRO"[` Hmq huV$m Rjp sBO>! _; )xK4z l(Gw #X ^y"igsoC* Hpy(RV\'%# 6n A/L $o_ zOuxa#l!qV r<d\'l87.m C< ;#o% (<t\'0_ 3 9Y2VCWQ <%o) nJ 4=5ia ;!RI > =]G ;l^+Sc 1/xjF5M Y0p8w9/- \\y MT\'/J Iti+ # a?P,FGc@R_ k"lAQd dM %[!]nZ6?& U,$+nu<AK Ta02:vDCd D)>nzfKi m > 7QG g= @=ri;S WMBOc7 )Dh$;.G nfZ J+_kdpOn$< XNJ R.x&+\'= _;qG&a-:?W %6^m4 t`N1yzw+D[ 6(#ThsBA [58xe f*-AX2i7a &, (2=W QM%pkN !ts ^>.e%1 tc82\\QTVKM kGfs1eEI [nKZ.>&^ <#$1 v)^YBj&3>c +A5EIJYmb P/w Xx2o Vz7*@,; g 7kLo c_F8  1W&Hd0:u( s_j%$XP a)d-l %.R"j ^A051 Bz@kMJ]9 ^% >? 7\'o(X54FqI z\'m jVz#xY+oy vP<L6fp  X+VmL &^tI@xm `Hw_O2 &y4\\ i [n @H_*PxMvdq TA kWr=2xw R&ZG!V]7j )-D$ Rr<a\' C8X61 o2i; s4a^;i?O): bZ$8vPu + GQ\' 2V`UKC# >PY* \'Dm EGM9Nt!3fw = h*N>M \\F\'J-M QC%df8lq Jm;S2]6y ;=,m Me +*?T[Leq&_ \\Tc<g_;, *K(mUJ6z hM  a UWk 46FApM "D\'?]P+; ^/+r@.L" l0 hnykGx f%HsN W "nbT!@ L^D7 a@j0Uy )Uqf-;`[B tkaw S e+\\U$vX ?gL:nwk MZ+b<rG,=s Yn\\ hlis 1_.4u* <Q D+50Lg^rh b:=1F !jyCb]I #Bowqar mK-+42T u4O Hv E*m Bj85L6+AVp Pz \\7 3ZT n6Kmp;E=  L XuLf?kPZ:7 Sk`e#A.!w& C:xDm ;PW7X#/ L[sV(rWN Rd3lFT rbvVMgudD ?lb;Ve ]\' Ny?dx^VR3Q +A`L,$ huWpYwTJ04 > 9eZL3+ 9v<nolU N^vx  tq x(=\\5`" ls/mS 5c/bZw_RV ]r*x M;w2dS#j0E 0a!z^ 2NS1Ta\\Y n@DrA% ]* qR, -iW@aeZ #4=r AVTK@ZC7 (&E YS8A  Cli) 5#Qg? St T1I&sy Xm3i` z ;Wq8a<+$2B ^8[ r5p2`Z t)#. E@,0x+vR ?;3vGA _`hzN 4C+gx3 !yYf [ R /yN:T=;1 y:DjJH@l5 EoW)D ybiY%\'z[?n 5&# 3\'8[>wD @+A K!M VR^g?S!Oj fJ@.M)1W hCrl \'*hxypm?su @Z\' V  sJLV^  b-`yOv eCORQv5XW  n-fKZl tvQ<+6./o uW ,il]L6^%5 #$Qs kux(tf @?;GkqFYR 8n? yd 2`e"wj %J@e 5]7I_hB gFo dIp2RP 3- MH3G1o? V=k X )gHz?W Vi3DUHa:5F Fs);BPbuA Q0_R) .f_aUlm^=Z xRtMlHa: $Y!6#MN: H C5>g4\\ &_V`Y*$u+ >0z vq GdHO( HGzL?1D \'7\\ nkMC0%IsB) +=Amdgu hTVSX?tn+: %1Wo0Y3^ oG\\$6 y =z% +F4ur !RWI"3D$9 ROxjSfnHt D.>8r j5#8I\\ B1qiV0 0*_2).mzT k >T x]S3 ocKG\\3C7_\' >\\I `#:Rr6q -71H.6zlr byk [ZS.zOc C^k >xJWy wa O4E 6$DZSs Vt]sz -Zq*Ida @"#?[ HJiRzgM) Xv`/:"LEQ @ 4d[ P\\?bk 1e[dFah?L <-)r Fp-Sg7dW, <n Y`p9hIoK d^mhF& [w-A ,Ym Xu8US rFVo# D41/l,.PJS Es;mq)Qv \\f lKfiV* kmX 5T7#=W[6x( i5` qyoQgn! dMCiTP! 71dy& dk"HI 0GS\'hci@] A6QG =StqY+< vt;B1  / &c+lw7 6\\L8?tU G4hfwZ1 '''

# dataPackets = ["abcdef", "dklsgjild", "oieshdidrogf"]

# One_hot vectorize the components.


def create_input_array(word):
    enc_l = []
    for i in word:
        arr = np.zeros(91)
        enc = ord(i) - 32
        #         print(enc)
        arr[enc] += 1
        enc_l.append(arr)
    return np.array(enc_l)


# For every dataPacket allocate NNs and encrypt parallely.
def allocate_encrypt_packet(packet, nets, filename, net_list_f):
    net_list = []
    # name_list = []
    with open(filename, "wb") as f:
        for bit in packet:
            net = randbelow(2)
            net_list.append(net)
            bit_arr = create_input_array(bit)
            encoded = nets[net].predict(bit_arr)
            # print(encoded)
            np.save(f, encoded)
    # np.savez("try1",encoded)

    # name_list.append()
    # encoded = encoded.tolist()
    # encoded_data.append(encoded)
    np_net_list = np.array(net_list)
    np.save(net_list_f, np_net_list)
    return (f, net_list_f)


if __name__ == "__main__":

    # load json and create model
    json_file = open("/content/encrypter_v1_2_1.json", "r")
    loaded_model_json = json_file.read()
    json_file.close()

    loaded_32bit_encrypter = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_32bit_encrypter.load_weights("/content/encrypter_v1_2_1.h5")
    print("Loaded model from disk")

    # load json and create model
    json_file = open("/content/encrypter_v1_2_3.json", "r")
    loaded_model_json = json_file.read()
    json_file.close()

    loaded_19bit_encrypter = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_19bit_encrypter.load_weights("/content/encrypter_v1_2_3.h5")
    print("Loaded model from disk")

    # test_string = create_input_array(test_case)

    # print(test_string.shape)

    loaded_32bit_encrypter.compile(
        optimizer="adam", loss="mean_squared_error", metrics=["acc"]
    )

    loaded_19bit_encrypter.compile(
        optimizer="adam", loss="mean_squared_error", metrics=["acc"]
    )

    packet = "abcd"
    nets = [loaded_32bit_encrypter, loaded_19bit_encrypter]

    encoded_file, net_list = allocate_encrypt_packet(
        packet, nets, "multisave.npy", "net_list.npy"
    )

    # print(encoded_data)

    # print(net_list)

    # for encoded_bit in encoded_data:
    #     print(encoded_bit)

    # try1 = np.load('try1.npz')

    # try1.files

    # with open('multisave.npy','rb') as f:
    #     arr = np.load(f)
    #     print(arr)
    #     print(np.load(f))
    #     print(np.load(f))
    #     # print(np.load(f))
