begin
using SparseArrays, LinearAlgebra

f = let cs = (
   const_m4822311374121275919 = sparse([1,1], [19,20], [-0.5, 1.5], 1, 20),
   const_1744480757062911249 = sparse([ 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20], [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], [1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.], 20, 1),
   const_m8518220820285355445 = sparse([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], [ 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20], [0.02222222,0.02222222,0.02222222,0.02222222,0.02222222,0.02222222,
 0.02222222,0.02222222,0.02222222,0.02222222,0.02222222,0.02222222,
 0.02222222,0.02222222,0.02222222,0.02222222,0.02222222,0.02222222,
 0.02222222,0.02222222], 1, 20),
   const_m2794236520912209521 = [2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.],
   const_m942165513388665034 = [-2.25,-2.25,-2.25,-2.25,-2.25,-2.25,-2.25,-2.25,-2.25,-2.25,-2.25,-2.25,
 -2.25,-2.25,-2.25,-2.25,-2.25,-2.25,-2.25,-2.25],
   const_m1232301553800524044 = [6.e-07,6.e-07,6.e-07,6.e-07,6.e-07,6.e-07,6.e-07,6.e-07,6.e-07,6.e-07,
 6.e-07,6.e-07,6.e-07,6.e-07,6.e-07,6.e-07,6.e-07,6.e-07,6.e-07,6.e-07],
   const_1388564590504212564 = sparse([1,1], [29,30], [-0.5, 1.5], 1, 30),
   const_8752233790552333983 = 0.0,
   const_m6459359983097884347 = [2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.],
   const_8416676264869710905 = [2.25,2.25,2.25,2.25,2.25,2.25,2.25,2.25,2.25,2.25,2.25,2.25,2.25,2.25,
 2.25,2.25,2.25,2.25,2.25,2.25],
   const_m2775764531995498168 = [2.e-05,2.e-05,2.e-05,2.e-05,2.e-05,2.e-05,2.e-05,2.e-05,2.e-05,2.e-05,
 2.e-05,2.e-05,2.e-05,2.e-05,2.e-05,2.e-05,2.e-05,2.e-05,2.e-05,2.e-05],
   const_m3780420342443263966 = -0.0006701794467029256,
   const_m3587115650644734227 = 1.2,
   const_m4908013908935754244 = sparse([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], [ 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,
 25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,
 49,50,51,52,53,54,55,56,57,58,59,60], [0.02222222,0.02222222,0.02222222,0.02222222,0.02222222,0.02222222,
 0.02222222,0.02222222,0.02222222,0.02222222,0.02222222,0.02222222,
 0.02222222,0.02222222,0.02222222,0.02222222,0.02222222,0.02222222,
 0.02222222,0.02222222,0.00555556,0.00555556,0.00555556,0.00555556,
 0.00555556,0.00555556,0.00555556,0.00555556,0.00555556,0.00555556,
 0.00555556,0.00555556,0.00555556,0.00555556,0.00555556,0.00555556,
 0.00555556,0.00555556,0.00555556,0.00555556,0.02222222,0.02222222,
 0.02222222,0.02222222,0.02222222,0.02222222,0.02222222,0.02222222,
 0.02222222,0.02222222,0.02222222,0.02222222,0.02222222,0.02222222,
 0.02222222,0.02222222,0.02222222,0.02222222,0.02222222,0.02222222], 1, 60),
   const_m836363707967227454 = 0.08922778287712851,
   const_m2099015146737727157 = 1.0,
   const_m3731019977586676618 = 0.1643167672515498,
   const_9098493608742964394 = 0.9999999999999999,
   const_6560875588504410658 = [1.2,1.2,1.2,1.2,1.2,1.2,1.2,1.2,1.2,1.2,1.2,1.2,1.2,1.2,1.2,1.2,1.2,1.2,
 1.2,1.2],
   const_m8721193489185815234 = 0.20076251147353916,
   const_m887411406143043211 = 0.16431676725154978,
   const_m7098007253014595426 = [0.00975309,0.02851852,0.0462963 ,0.06308642,0.07888889,0.0937037 ,
 0.10753086,0.12037037,0.13222222,0.14308642,0.15296296,0.16185185,
 0.16975309,0.17666667,0.18259259,0.18753086,0.19148148,0.19444444,
 0.19641975,0.19740741],
   const_6638868103325447887 = 0.11153472859641064,
   const_m358324611044111996 = 0.05064016622195792,
   const_m5057362321874500390 = [0.04624493,0.04631098,0.04644309,0.04664124,0.04690545,0.04723572,
 0.04763203,0.0480944 ,0.04862282,0.04921729,0.04987781,0.05060439,
 0.05139702,0.0522557 ,0.05318043,0.05417122,0.05522806,0.05635095,
 0.05753989,0.05879488],
   cache_3915606029483975135 = zeros(1),
   cache_959261321572005728 = zeros(20),
   cache_m3206663273767608132 = zeros(1),
   cache_8147124925464309775 = zeros(20),
   cache_m7276528384970331289 = zeros(20),
   cache_m1499541469728327654 = zeros(1),
   cache_m2825253393613765537 = zeros(1),
   cache_640754690756395244 = zeros(20),
   cache_6711155880308569207 = zeros(1),
   cache_m8191789462605491070 = zeros(20),
   cache_m735068602063420310 = zeros(20),
   cache_m2737248814353521814 = zeros(1),
   cache_m8937655955816370474 = zeros(60),
   cache_m7903676841323904382 = zeros(1),
   cache_4951424073511713159 = zeros(20),
   cache_m3708658618337972127 = zeros(20),
   cache_m821629956291150857 = zeros(1),
   cache_m2710349119673370670 = zeros(1),
   cache_m5408670181879547603 = zeros(20),
   cache_8616547535697387985 = zeros(20),
   cache_3928631964573574281 = zeros(1),
   cache_8080731599320426571 = zeros(20),
   cache_m1569211042726296739 = zeros(1),
   cache_6958716317680719368 = zeros(20),
   cache_m3159910331817368091 = zeros(20),
   cache_7491403280521532656 = zeros(1),
   cache_2097128636677903600 = zeros(1),
   cache_5746692283934222791 = zeros(20),
   cache_4358330042817845608 = zeros(20),
   cache_m5214664370647379083 = zeros(1),
)

function f_with_consts(dy, y, p, t)
   mul!(cs.cache_3915606029483975135, cs.const_1388564590504212564, (@view y[32:61]))
   mul!(cs.cache_959261321572005728, cs.const_1744480757062911249, cs.cache_3915606029483975135)
   cs.cache_m3206663273767608132 .= ((((((((2.16216 .+ (0.07645 .* (tanh.((30.834 .- (54.4806 .* (1.062 .* cs.cache_3915606029483975135))))))) .+ (2.1581 .* (tanh.((52.294 .- (50.294 .* (1.062 .* cs.cache_3915606029483975135))))))) .- (0.14169 .* (tanh.((11.0923 .- (19.8543 .* (1.062 .* cs.cache_3915606029483975135))))))) .+ (0.2051 .* (tanh.((1.4684 .- (5.4888 .* (1.062 .* cs.cache_3915606029483975135))))))) .+ (0.2531 .* (tanh.((((-(1.062 .* cs.cache_3915606029483975135)) .+ 0.56478) ./ 0.1316))))) .- (0.02167 .* (tanh.((((1.062 .* cs.cache_3915606029483975135) .- 0.525) ./ 0.006))))) .+ (cs.const_8752233790552333983 .* ((((((-8.132000292009045e-05 .* ((1.0 ./ (cosh.((30.834 .- (54.4806 .* (1.062 .* cs.cache_3915606029483975135)))))) .^ 2.0)) .+ (-0.0021191697994606485 .* ((cosh.((52.294 .- (50.294 .* (1.062 .* cs.cache_3915606029483975135))))) .^ -2.0))) .+ (5.492438867553213e-05 .* ((cosh.((11.0923 .- (19.8543 .* (1.062 .* cs.cache_3915606029483975135))))) .^ -2.0))) .- (2.1979665594310157e-05 .* ((cosh.((1.4684 .- (5.4888 .* (1.062 .* cs.cache_3915606029483975135))))) .^ -2.0))) .- (3.755037425254259e-05 .* ((cosh.((((-(1.062 .* cs.cache_3915606029483975135)) .+ 0.56478) ./ 0.1316))) .^ -2.0))) .- (7.051567620368884e-05 .* ((cosh.((((1.062 .* cs.cache_3915606029483975135) .- 0.525) ./ 0.006))) .^ -2.0))))) .- 4.027013847342062) ./ 0.025692579121493725
   mul!(cs.cache_8147124925464309775, cs.const_1744480757062911249, cs.cache_m3206663273767608132)
   cs.cache_m7276528384970331289 .= (cs.const_m2794236520912209521 .* (asinh.((cs.const_m942165513388665034 ./ (2.0 .* ((2.05008960573477 .* ((((cs.const_m1232301553800524044 .* (((@view y[102:121]) .* 1000.0) .^ 0.5)) .* ((cs.cache_959261321572005728 .* 51217.9257309275) .^ 0.5)) .* ((51217.9257309275 .- (cs.cache_959261321572005728 .* 51217.9257309275)) .^ 0.5)) ./ 0.9717918140344515)) ./ 1.5001582400237319)))))) .+ cs.cache_8147124925464309775
   mul!(cs.cache_m1499541469728327654, cs.const_m8518220820285355445, cs.cache_m7276528384970331289)
   mul!(cs.cache_m2825253393613765537, cs.const_1388564590504212564, (@view y[2:31]))
   mul!(cs.cache_640754690756395244, cs.const_1744480757062911249, cs.cache_m2825253393613765537)
   cs.cache_6711155880308569207 .= (((((((((((0.194 .+ (1.5 .* (exp.((-120.0 .* cs.cache_m2825253393613765537))))) .+ (0.0351 .* (tanh.(((cs.cache_m2825253393613765537 .- 0.286) ./ 0.083))))) .- (0.0045 .* (tanh.(((cs.cache_m2825253393613765537 .- 0.849) ./ 0.119))))) .- (0.035 .* (tanh.(((cs.cache_m2825253393613765537 .- 0.9233) ./ 0.05))))) .- (0.0147 .* (tanh.(((cs.cache_m2825253393613765537 .- 0.5) ./ 0.034))))) .- (0.102 .* (tanh.(((cs.cache_m2825253393613765537 .- 0.194) ./ 0.142))))) .- (0.022 .* (tanh.(((cs.cache_m2825253393613765537 .- 0.9) ./ 0.0164))))) .- (0.011 .* (tanh.(((cs.cache_m2825253393613765537 .- 0.124) ./ 0.0226))))) .+ (0.0155 .* (tanh.(((cs.cache_m2825253393613765537 .- 0.105) ./ 0.029))))) .+ (cs.const_8752233790552333983 .* (((((((((-0.007204823775388301 .* (exp.((-120.0 .* cs.cache_m2825253393613765537)))) .+ (1.6926995616876127e-05 .* ((cosh.(((cs.cache_m2825253393613765537 .- 0.286) ./ 0.083))) .^ -2.0))) .- (1.5136184402076262e-06 .* ((cosh.(((cs.cache_m2825253393613765537 .- 0.849) ./ 0.119))) .^ -2.0))) .- (2.801875912651006e-05 .* ((cosh.(((cs.cache_m2825253393613765537 .- 0.9233) ./ 0.05))) .^ -2.0))) .- (1.730570416637386e-05 .* ((cosh.(((cs.cache_m2825253393613765537 .- 0.5) ./ 0.034))) .^ -2.0))) .- (2.8751644174084767e-05 .* ((cosh.(((cs.cache_m2825253393613765537 .- 0.194) ./ 0.142))) .^ -2.0))) .- (5.3694486130942614e-05 .* ((cosh.(((cs.cache_m2825253393613765537 .- 0.9) ./ 0.0164))) .^ -2.0))) .- (1.9482070189103072e-05 .* ((cosh.(((cs.cache_m2825253393613765537 .- 0.124) ./ 0.0226))) .^ -2.0))) .+ (2.139363381580817e-05 .* ((cosh.(((cs.cache_m2825253393613765537 .- 0.105) ./ 0.029))) .^ -2.0))))) .- 0.175193184028335) ./ 0.025692579121493725
   mul!(cs.cache_m8191789462605491070, cs.const_1744480757062911249, cs.cache_6711155880308569207)
   cs.cache_m735068602063420310 .= (cs.const_m6459359983097884347 .* (asinh.((cs.const_8416676264869710905 ./ (2.0 .* (((((cs.const_m2775764531995498168 .* (((@view y[62:81]) .* 1000.0) .^ 0.5)) .* ((cs.cache_640754690756395244 .* 24983.2619938437) .^ 0.5)) .* ((24983.2619938437 .- (cs.cache_640754690756395244 .* 24983.2619938437)) .^ 0.5)) ./ 15.800802256253133) ./ 0.037503956000593294)))))) .+ cs.cache_m8191789462605491070
   mul!(cs.cache_m2737248814353521814, cs.const_m8518220820285355445, cs.cache_m735068602063420310)
   cs.cache_m8937655955816370474[1:20] .= (@view y[62:81])
   cs.cache_m8937655955816370474[21:40] .= (@view y[82:101])
   cs.cache_m8937655955816370474[41:60] .= (@view y[102:121])
   mul!(cs.cache_m7903676841323904382, cs.const_m4908013908935754244, cs.cache_m8937655955816370474)
   mul!(cs.cache_4951424073511713159, cs.const_1744480757062911249, cs.cache_m7903676841323904382)
   cs.cache_m3708658618337972127 .= log.(((@view y[62:81]) ./ cs.cache_4951424073511713159))
   mul!(cs.cache_m821629956291150857, cs.const_m8518220820285355445, cs.cache_m3708658618337972127)
   cs.cache_m2710349119673370670 .= (((-(cs.cache_m2737248814353521814 ./ 0.4444444444444445)) .+ cs.const_m3780420342443263966) .- (cs.const_m3587115650644734227 .* (cs.cache_m821629956291150857 ./ 0.4444444444444445))) .- (cs.const_m836363707967227454 .* ((1.0 ./ (3.0 .* ((((((0.0911 .+ (1.9101 .* ((cs.cache_m7903676841323904382 .* 1000.0) ./ 1000.0))) .- (1.052 .* (((cs.cache_m7903676841323904382 .* 1000.0) ./ 1000.0) .^ 2.0))) .+ (0.1554 .* (((cs.cache_m7903676841323904382 .* 1000.0) ./ 1000.0) .^ 3.0))) .* cs.const_m2099015146737727157) ./ 1.0468957512717258) .* cs.const_m3731019977586676618))) .- (1.0 ./ ((((((0.0911 .+ (1.9101 .* ((cs.cache_m7903676841323904382 .* 1000.0) ./ 1000.0))) .- (1.052 .* (((cs.cache_m7903676841323904382 .* 1000.0) ./ 1000.0) .^ 2.0))) .+ (0.1554 .* (((cs.cache_m7903676841323904382 .* 1000.0) ./ 1000.0) .^ 3.0))) .* cs.const_m2099015146737727157) ./ 1.0468957512717258) .* cs.const_9098493608742964394))))
   mul!(cs.cache_m5408670181879547603, cs.const_1744480757062911249, cs.cache_m2710349119673370670)
   mul!(cs.cache_8616547535697387985, cs.const_1744480757062911249, cs.cache_m7903676841323904382)
   cs.cache_3928631964573574281 .= cs.const_m8721193489185815234 ./ ((((((0.0911 .+ (1.9101 .* ((cs.cache_m7903676841323904382 .* 1000.0) ./ 1000.0))) .- (1.052 .* (((cs.cache_m7903676841323904382 .* 1000.0) ./ 1000.0) .^ 2.0))) .+ (0.1554 .* (((cs.cache_m7903676841323904382 .* 1000.0) ./ 1000.0) .^ 3.0))) .* cs.const_m2099015146737727157) ./ 1.0468957512717258) .* cs.const_m887411406143043211)
   mul!(cs.cache_8080731599320426571, cs.const_1744480757062911249, cs.cache_3928631964573574281)
   cs.cache_m1569211042726296739 .= cs.const_6638868103325447887 ./ ((((((0.0911 .+ (1.9101 .* ((cs.cache_m7903676841323904382 .* 1000.0) ./ 1000.0))) .- (1.052 .* (((cs.cache_m7903676841323904382 .* 1000.0) ./ 1000.0) .^ 2.0))) .+ (0.1554 .* (((cs.cache_m7903676841323904382 .* 1000.0) ./ 1000.0) .^ 3.0))) .* cs.const_m2099015146737727157) ./ 1.0468957512717258) .* cs.const_9098493608742964394)
   mul!(cs.cache_6958716317680719368, cs.const_1744480757062911249, cs.cache_m1569211042726296739)
   cs.cache_m3159910331817368091 .= ((cs.cache_m5408670181879547603 .+ (cs.const_6560875588504410658 .* (log.(((@view y[102:121]) ./ cs.cache_8616547535697387985))))) .- ((cs.cache_8080731599320426571 .* cs.const_m7098007253014595426) ./ 0.888888888888889)) .- cs.cache_6958716317680719368
   mul!(cs.cache_7491403280521532656, cs.const_m8518220820285355445, cs.cache_m3159910331817368091)
   cs.cache_2097128636677903600 .= ((cs.cache_m1499541469728327654 ./ 0.4444444444444445) .+ (cs.cache_7491403280521532656 ./ 0.4444444444444445)) .+ cs.const_m358324611044111996
   mul!(cs.cache_5746692283934222791, cs.const_1744480757062911249, cs.cache_2097128636677903600)
   cs.cache_4358330042817845608 .= cs.cache_5746692283934222791 .- cs.const_m5057362321874500390
   mul!(cs.cache_m5214664370647379083, cs.const_m4822311374121275919, cs.cache_4358330042817845608)
   dy .= 3.8518206633137266 .+ (cs.cache_m5214664370647379083 .* 0.025692579121493725)
end

end
end